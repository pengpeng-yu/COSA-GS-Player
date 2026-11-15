#include "network.h"
#include "gelu_lut.h"

#include <algorithm>
#include <cstring>
#include <limits>

namespace cosa {
namespace {

// Match CUDA's narrowing to the low 32 bits without signed-overflow UB.
int32_t narrowI32(int64_t value) {
    const uint32_t bits = static_cast<uint32_t>(value);
    int32_t result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

int64_t roundedShift(int64_t value, int shift) {
    if (shift == 0) return value;
    const uint64_t magnitude = value < 0
        ? uint64_t(-(value + 1)) + 1 : uint64_t(value);
    const uint64_t mask = (uint64_t(1) << shift) - 1;
    const uint64_t rounded = (magnitude >> shift) +
        ((magnitude & mask) >= (uint64_t(1) << (shift - 1)));
    return value < 0 ? -int64_t(rounded) : int64_t(rounded);
}

int32_t gelu(int32_t input) {
    if (input >= 6 * FIXED_ONE) return input;
    if (input <= -6 * FIXED_ONE) return 0;

    const int32_t magnitude = input < 0 ? -input : input;
    constexpr int FractionBits = 11;
    const int index = magnitude >> FractionBits;
    const int fraction = magnitude & ((1 << FractionBits) - 1);
    const int64_t value = GELU_LUT[index];
    const int64_t numerator = (value << FractionBits) +
        (int64_t(GELU_LUT[index + 1]) - value) * fraction;
    return std::max(input, int32_t(0)) - int32_t(roundedShift(numerator, 15));
}

int8_t clampI8(int64_t value) {
    return int8_t(std::clamp(value, int64_t(-127), int64_t(127)));
}

void validateMultiplier(uint32_t multiplier) {
    require(multiplier > 0 && multiplier < (uint32_t(1) << 31),
            "Invalid integer network multiplier");
}

void validateShift(int32_t shift) {
    require(shift >= 0 && shift <= 62, "Invalid integer network shift");
}

} // namespace

int64_t roundDivide(int64_t value, int64_t positiveDenominator) {
    require(positiveDenominator > 0, "Integer rounding requires a positive denominator");
    const uint64_t denominator = uint64_t(positiveDenominator);
    const uint64_t magnitude = value < 0
        ? uint64_t(-(value + 1)) + 1 : uint64_t(value);
    const uint64_t rounded = magnitude / denominator +
        (magnitude % denominator >= (denominator + 1) / 2);
    if (value >= 0) return int64_t(rounded);
    return rounded == 0 ? 0 : -int64_t(rounded - 1) - 1;
}

int32_t checkedI32(int64_t value) {
    require(value >= std::numeric_limits<int32_t>::min() &&
            value <= std::numeric_limits<int32_t>::max(),
            "Fixed-point value exceeds int32");
    return int32_t(value);
}

IntegerNetwork IntegerNetwork::read(Reader& reader, const std::vector<int>& widths,
                                    bool finalGelu) {
    require(widths.size() >= 2, "Integer network requires a Linear layer");
    for (const int width : widths)
        require(width > 0 && width <= 4096, "Unsupported integer network width");

    IntegerNetwork network;
    network.input_.multiplier = reader.u32();
    network.input_.shift = reader.i32();
    network.input_.zeroPoint = reader.i32();
    validateMultiplier(network.input_.multiplier);
    validateShift(network.input_.shift);

    for (size_t i = 1; i < widths.size(); ++i) {
        Layer layer;
        layer.inputWidth = size_t(widths[i - 1]);
        layer.outputWidth = size_t(widths[i]);
        layer.weights = reader.array<int8_t>(layer.inputWidth * layer.outputWidth);
        layer.bias = reader.array<int32_t>(layer.outputWidth);
        layer.multipliers = reader.array<uint32_t>(layer.outputWidth);
        layer.shift = reader.i32();
        layer.output.multiplier = reader.u32();
        layer.output.shift = reader.i32();
        layer.output.zeroPoint = reader.i32();
        layer.gelu = i + 1 < widths.size() || finalGelu;
        for (const uint32_t multiplier : layer.multipliers)
            validateMultiplier(multiplier);
        validateShift(layer.shift);
        validateMultiplier(layer.output.multiplier);
        validateShift(layer.output.shift);
        network.layers_.push_back(std::move(layer));
    }
    return network;
}

IntMatrix IntegerNetwork::forward(const IntMatrix& input) const {
    require(!layers_.empty(), "Integer network has no layers");
    require(input.cols == layers_.front().inputWidth &&
            input.values.size() == input.rows * input.cols,
            "Integer network input shape mismatch");

    size_t scratchWidth = input.cols;
    for (const Layer& layer : layers_)
        scratchWidth = std::max(scratchWidth, layer.outputWidth);
    std::vector<int8_t> current(scratchWidth);
    std::vector<int8_t> next(scratchWidth);
    IntMatrix output(input.rows, layers_.back().outputWidth);

    // Networks are independent across anchors. Keep hidden activations in two
    // small row buffers instead of allocating full N x hidden-width tensors.
    for (size_t row = 0; row < input.rows; ++row) {
        for (size_t channel = 0; channel < input.cols; ++channel) {
            const int32_t value = input.values[row * input.cols + channel];
            current[channel] = clampI8(
                roundedShift(int64_t(value) * input_.multiplier, input_.shift) +
                input_.zeroPoint);
        }

        for (size_t layerIndex = 0; layerIndex < layers_.size(); ++layerIndex) {
            const Layer& layer = layers_[layerIndex];
            const bool intermediate = layerIndex + 1 < layers_.size();
            for (size_t channel = 0; channel < layer.outputWidth; ++channel) {
                const int8_t* weights = layer.weights.data() + channel * layer.inputWidth;
                int32_t dot = 0;
                // The checked width bound keeps the dot product below int32.
                // LLVM can vectorize this exact integer loop with -msimd128.
                for (size_t k = 0; k < layer.inputWidth; ++k)
                    dot += int32_t(current[k]) * int32_t(weights[k]);
                const int32_t biased = narrowI32(int64_t(dot) + layer.bias[channel]);
                int64_t value = roundedShift(
                    int64_t(biased) * layer.multipliers[channel], layer.shift);
                if (layer.gelu) value = gelu(narrowI32(value));
                if (intermediate) {
                    value = roundedShift(value * layer.output.multiplier, layer.output.shift);
                    next[channel] = clampI8(value + layer.output.zeroPoint);
                } else {
                    // CUDA ignores zero_point_out for GELU-to-int32.
                    if (!layer.gelu) value += layer.output.zeroPoint;
                    output.values[row * output.cols + channel] = narrowI32(value);
                }
            }
            if (intermediate) current.swap(next);
        }
    }
    return output;
}

std::vector<uint32_t> IntegerNetwork::gpuParameters() const {
    require(!layers_.empty() && layers_.size() <= 2, "GPU network requires one or two layers");
    std::vector<uint32_t> words{
        static_cast<uint32_t>(layers_.size()), input_.multiplier,
        static_cast<uint32_t>(input_.shift), static_cast<uint32_t>(input_.zeroPoint),
        0, 0, static_cast<uint32_t>(inputWidth()), static_cast<uint32_t>(outputWidth())
    };
    for (size_t index = 0; index < layers_.size(); ++index) {
        const auto& layer = layers_[index];
        require(layer.inputWidth <= 64 && layer.outputWidth <= 64, "GPU network width exceeds 64");
        words[4 + index] = static_cast<uint32_t>(words.size());
        const uint32_t header[]{
            static_cast<uint32_t>(layer.inputWidth), static_cast<uint32_t>(layer.outputWidth),
            static_cast<uint32_t>(layer.gelu), static_cast<uint32_t>(layer.shift),
            layer.output.multiplier, static_cast<uint32_t>(layer.output.shift),
            static_cast<uint32_t>(layer.output.zeroPoint), 0
        };
        words.insert(words.end(), std::begin(header), std::end(header));
        const size_t rowWords = (layer.inputWidth + 3) / 4;
        for (size_t row = 0; row < layer.outputWidth; ++row) {
            for (size_t word = 0; word < rowWords; ++word) {
                uint32_t packed = 0;
                for (size_t byte = 0; byte < 4; ++byte) {
                    const size_t column = word * 4 + byte;
                    if (column < layer.inputWidth)
                        packed |= uint32_t(uint8_t(layer.weights[row * layer.inputWidth + column])) << (8 * byte);
                }
                words.push_back(packed);
            }
        }
        for (const auto value : layer.bias) words.push_back(static_cast<uint32_t>(value));
        words.insert(words.end(), layer.multipliers.begin(), layer.multipliers.end());
    }
    return words;
}

} // namespace cosa
