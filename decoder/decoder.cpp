#include "common.h"
#include "entropy.h"
#include "network.h"
#include "gelu_lut.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <variant>
#include <emscripten/emscripten.h>

namespace cosa {

constexpr size_t FEATURE_COUNT = 32;
constexpr size_t OFFSETS_PER_ANCHOR = 10;
constexpr size_t RENDER_WEIGHTS_PER_HEAD = 3496;

using Storage = std::variant<std::vector<int16_t>, std::vector<int32_t>,
                             std::vector<uint32_t>, std::vector<float>>;

struct Output {
    std::string name;
    Storage values;
};

struct Result {
    std::vector<Output> arrays;
    size_t anchorCount = 0;
    size_t activeOffsetCount = 0;
    float distanceScale = 1;

    template<class T>
    void add(std::string name, size_t rows, size_t cols, std::vector<T> values) {
        require(values.size() == rows * cols, "Invalid output dimensions: " + name);
        arrays.push_back({std::move(name), std::move(values)});
    }
};

Result result;
std::string lastError;

struct Step {
    uint64_t multiplier;
    float floating;
    uint32_t shift;

    static Step read(Reader& reader) {
        const uint64_t multiplier = reader.u64();
        const uint32_t shift = reader.u32();
        require(multiplier > 0 && multiplier < (uint64_t{1} << 47) && shift <= 62,
                "Invalid reconstruction step");
        return {multiplier,
                static_cast<float>(std::ldexp(static_cast<double>(multiplier), -static_cast<int>(shift) - 20)),
                shift};
    }
};

float halfFloat(uint16_t bits) {
    const int exponent = (bits >> 10) & 31;
    const int fraction = bits & 1023;
    require(exponent != 31, "Non-finite render network parameter");
    const float value = exponent ? std::ldexp(static_cast<float>(1024 + fraction), exponent - 25)
                                 : std::ldexp(static_cast<float>(fraction), -24);
    return bits & 32768 ? -value : value;
}

std::vector<float> readRenderWeights(Reader& reader) {
    std::vector<float> weights(3 * RENDER_WEIGHTS_PER_HEAD, 0);
    // Serialized order is opacity/cov/color. WGSL order is opacity/color/cov.
    constexpr size_t destinations[]{0, 2, 1};
    constexpr size_t inputs[]{36, 36, 35};
    constexpr size_t outputs[]{10, 70, 30};

    for (size_t head = 0; head < 3; ++head) {
        float* target = weights.data() + destinations[head] * RENDER_WEIGHTS_PER_HEAD;
        for (size_t output = 0; output < 32; ++output)
            for (size_t input = 0; input < inputs[head]; ++input)
                target[input * 32 + output] = halfFloat(reader.u16());
        for (size_t i = 0; i < 32; ++i) target[1152 + i] = halfFloat(reader.u16());
        for (size_t i = 0; i < outputs[head] * 32; ++i) target[1184 + i] = halfFloat(reader.u16());
        for (size_t i = 0; i < outputs[head]; ++i) target[3424 + i] = halfFloat(reader.u16());
    }
    return weights;
}

struct ParsedScene {
    std::array<int32_t, 3> origin;
    std::array<int32_t, 3> shape;
    double gridSize;
    float distanceScale;
    std::vector<IntegerNetwork> networks;
    std::vector<float> weights;
    std::array<Step, 5> steps;
    std::vector<uint16_t> maskCdf;
    std::array<Bytes, 10> streams;
    GaussianTables tables;

    ParsedScene(Bytes encoded, Bytes tableBytes, const int32_t* config, size_t configCount,
                double gridSizeValue) : gridSize(gridSizeValue) {
        constexpr int32_t expected[]{32, 10, 24, 24, 1, 4, 0, 1, 1, 0};
        require(configCount == 10, "Expected ten codec configuration values");
        for (size_t i = 0; i < configCount; ++i)
            require(i == 6 ? (config[i] == 0 || config[i] == 1) : config[i] == expected[i],
                    "Unsupported codec architecture at configuration index " + std::to_string(i));

        Reader reader(encoded);
        for (auto& value : origin) value = reader.i32();
        for (auto& value : shape) {
            value = reader.i32();
            require(value > 1, "Invalid coordinate shape");
        }
        if (config[6]) {
            reader.f64();
            gridSize = reader.f64();
        }
        require(std::isfinite(gridSize) && gridSize > 0, "Invalid anchor grid size");
        distanceScale = reader.f32();
        require(std::isfinite(distanceScale) && distanceScale > 0, "Invalid distance scale");

        for (int stage = 0; stage < 4; ++stage)
            networks.push_back(IntegerNetwork::read(reader, {24 + stage, 24, 2}));
        networks.push_back(IntegerNetwork::read(reader, {4, 24}, true));
        networks.push_back(IntegerNetwork::read(reader, {48, 24}, true));
        networks.push_back(IntegerNetwork::read(reader, {3, 24, 24}, true));
        networks.push_back(IntegerNetwork::read(reader, {1, 24}, true));
        networks.push_back(IntegerNetwork::read(reader, {24, 24, 24}));
        networks.push_back(IntegerNetwork::read(reader, {24, 24, 24}));
        networks.push_back(IntegerNetwork::read(reader, {24, 60, 60}));
        networks.push_back(IntegerNetwork::read(reader, {24, 24, 2}));
        networks.push_back(IntegerNetwork::read(reader, {24, 24, 6}));
        networks.push_back(IntegerNetwork::read(reader, {24, 64, 64}));

        weights = readRenderWeights(reader);
        for (auto& step : steps) step = Step::read(reader);
        maskCdf = reader.array<uint16_t>((1 << OFFSETS_PER_ANCHOR) - 1);
        for (auto& stream : streams) {
            const auto size = reader.u32();
            stream = reader.take(size);
        }
        require(reader.remaining() == 0, "Trailing integer scene bytes");
        tables = GaussianTables::load(tableBytes);
    }
};

IntMatrix normalizedCoordinates(const IntMatrix& coordinates, const std::array<int32_t, 3>& shape) {
    IntMatrix normalized(coordinates.rows, 3);
    for (size_t i = 0; i < normalized.values.size(); ++i) {
        const auto coordinate = coordinates.values[i];
        require(coordinate >= 0 && coordinate < shape[i % 3], "Coordinate exceeds scene shape");
        normalized.values[i] = checkedI32(
            roundDivide(int64_t{coordinate} * 2 * FIXED_ONE, shape[i % 3] - 1) - FIXED_ONE);
    }
    return normalized;
}

std::vector<float> anchorPositions(const IntMatrix& coordinates, const ParsedScene& model) {
    std::vector<float> positions(coordinates.rows * 3);
    for (size_t row = 0; row < coordinates.rows; ++row)
        for (size_t axis = 0; axis < 3; ++axis)
            positions[row * 3 + axis] =
                static_cast<float>(int64_t{coordinates.values[row * 3 + axis]} + model.origin[axis]) *
                static_cast<float>(model.gridSize);
    return positions;
}

std::vector<uint32_t> activeOffsetSlots(const std::vector<int16_t>& symbols) {
    std::vector<uint32_t> slots;
    for (size_t row = 0; row < symbols.size(); ++row)
        for (size_t slot = 0; slot < OFFSETS_PER_ANCHOR; ++slot)
            if ((symbols[row] >> (OFFSETS_PER_ANCHOR - 1 - slot)) & 1)
                slots.push_back(static_cast<uint32_t>(row * OFFSETS_PER_ANCHOR + slot));
    require(!slots.empty(), "Scene contains no active Gaussians");
    return slots;
}

struct HybridSession {
    std::unique_ptr<ParsedScene> model;
    std::vector<std::vector<uint32_t>> parameters;
    std::vector<int16_t> symbols;
    IntMatrix reference;
};
HybridSession hybrid;

void openHybrid(Bytes encoded, Bytes tableBytes, const int32_t* config, size_t configCount, double gridSize) {
    hybrid.model = std::make_unique<ParsedScene>(encoded, tableBytes, config, configCount, gridSize);
    const auto& model = *hybrid.model;
    result.distanceScale = model.distanceScale;
    auto coordinates = decodeOctree(model.streams[0]);
    require(coordinates.rows > 0 && coordinates.rows <= UINT32_MAX / OFFSETS_PER_ANCHOR,
            "Invalid anchor count");
    result.anchorCount = coordinates.rows;
    const size_t count = result.anchorCount;
    auto normalized = normalizedCoordinates(coordinates, model.shape);
    auto positions = anchorPositions(coordinates, model);
    auto mask = decodeMask(model.streams[5], model.maskCdf, count);
    auto slots = activeOffsetSlots(mask);
    result.activeOffsetCount = slots.size();
    for (const auto& network : model.networks) hybrid.parameters.push_back(network.gpuParameters());

    result.add("positions", count, 3, std::move(positions));
    result.add("normalized", count, 3, std::move(normalized.values));
    result.add("activeSlots", result.activeOffsetCount, 1, std::move(slots));
    result.add("weights", 3, RENDER_WEIGHTS_PER_HEAD, std::move(hybrid.model->weights));
    result.add("gelu", 3073, 1, std::vector<int32_t>(std::begin(GELU_LUT), std::end(GELU_LUT)));
    result.add("anchor_coords", count, 3, std::move(coordinates.values));
    result.add("offset_mask_symbols", count, 1, std::move(mask));
}

} // namespace cosa

extern "C" {

EMSCRIPTEN_KEEPALIVE const char* codec_error() { return cosa::lastError.c_str(); }
EMSCRIPTEN_KEEPALIVE uint32_t codec_array_count() { return cosa::result.arrays.size(); }
EMSCRIPTEN_KEEPALIVE const char* codec_array_name(uint32_t index) {
    return cosa::result.arrays.at(index).name.c_str();
}

EMSCRIPTEN_KEEPALIVE const void* codec_array_data(uint32_t index) {
    return std::visit(
        [](const auto& values) -> const void* { return values.data(); },
        cosa::result.arrays.at(index).values);
}

EMSCRIPTEN_KEEPALIVE uint32_t codec_array_bytes(uint32_t index) {
    return std::visit(
        [](const auto& values) -> uint32_t { return values.size() * sizeof(values[0]); },
        cosa::result.arrays.at(index).values);
}

EMSCRIPTEN_KEEPALIVE uint32_t codec_anchor_count() { return cosa::result.anchorCount; }
EMSCRIPTEN_KEEPALIVE uint32_t codec_active_count() { return cosa::result.activeOffsetCount; }
EMSCRIPTEN_KEEPALIVE float codec_distance_scale() { return cosa::result.distanceScale; }

EMSCRIPTEN_KEEPALIVE void codec_release() {
    cosa::result = {};
    cosa::hybrid = {};
}

EMSCRIPTEN_KEEPALIVE int hybrid_open(const uint8_t* bytes, uint32_t byteCount,
    const uint8_t* tables, uint32_t tableBytes, const int32_t* config,
    uint32_t configCount, double gridSize) {
    codec_release();
    cosa::lastError.clear();
    try {
        cosa::openHybrid({bytes, byteCount}, {tables, tableBytes}, config, configCount, gridSize);
        return 0;
    } catch (const std::exception& error) {
        cosa::lastError = error.what();
        codec_release();
        return -1;
    }
}

EMSCRIPTEN_KEEPALIVE void hybrid_clear_arrays() { cosa::result.arrays.clear(); }
EMSCRIPTEN_KEEPALIVE uint32_t hybrid_network_count() { return cosa::hybrid.parameters.size(); }

EMSCRIPTEN_KEEPALIVE const uint32_t* hybrid_network_data(uint32_t index) {
    return cosa::hybrid.parameters.at(index).data();
}

EMSCRIPTEN_KEEPALIVE uint32_t hybrid_network_bytes(uint32_t index) {
    return cosa::hybrid.parameters.at(index).size() * sizeof(uint32_t);
}

EMSCRIPTEN_KEEPALIVE uint32_t hybrid_step_mul_lo(uint32_t index) {
    return static_cast<uint32_t>(cosa::hybrid.model->steps.at(index).multiplier);
}

EMSCRIPTEN_KEEPALIVE uint32_t hybrid_step_mul_hi(uint32_t index) {
    return static_cast<uint32_t>(cosa::hybrid.model->steps.at(index).multiplier >> 32);
}

EMSCRIPTEN_KEEPALIVE uint32_t hybrid_step_shift(uint32_t index) {
    return cosa::hybrid.model->steps.at(index).shift;
}

EMSCRIPTEN_KEEPALIVE float hybrid_step_float(uint32_t index) {
    return cosa::hybrid.model->steps.at(index).floating;
}

EMSCRIPTEN_KEEPALIVE int hybrid_entropy(uint32_t stream, const uint8_t* cdfIndices, uint32_t count) {
    cosa::lastError.clear();
    try {
        cosa::require(bool(cosa::hybrid.model), "No open hybrid decoder");
        const bool latent = stream >= 1 && stream <= 4;
        cosa::require(latent || (stream >= 6 && stream <= 9), "Invalid hybrid entropy stream");
        const size_t expected = latent || stream == 6 ? cosa::result.anchorCount
            : stream == 7 ? cosa::result.activeOffsetCount * 3
            : stream == 8 ? cosa::result.anchorCount * 3
                          : cosa::result.anchorCount * cosa::FEATURE_COUNT;
        cosa::require(count == expected, "Hybrid entropy symbol count mismatch");
        const auto& model = *cosa::hybrid.model;
        cosa::decodeGaussianInto(model.streams[stream], cdfIndices, count, model.tables, cosa::hybrid.symbols);
        return 0;
    } catch (const std::exception& error) {
        cosa::lastError = error.what();
        cosa::hybrid.symbols.clear();
        return -1;
    }
}

EMSCRIPTEN_KEEPALIVE const int16_t* hybrid_symbols_data() { return cosa::hybrid.symbols.data(); }
EMSCRIPTEN_KEEPALIVE uint32_t hybrid_symbols_count() { return cosa::hybrid.symbols.size(); }
EMSCRIPTEN_KEEPALIVE uint32_t hybrid_symbols_bytes() { return cosa::hybrid.symbols.size() * sizeof(int16_t); }

// Explicit diagnostics only: evaluate the same input with the exact CPU network.
EMSCRIPTEN_KEEPALIVE const int32_t* hybrid_network_reference(
    uint32_t index, const int32_t* input, uint32_t rows) {
    cosa::lastError.clear();
    cosa::hybrid.reference = {};
    try {
        cosa::require(bool(cosa::hybrid.model), "No open hybrid decoder");
        const auto& network = cosa::hybrid.model->networks.at(index);
        cosa::IntMatrix values(rows, network.inputWidth());
        std::copy_n(input, values.values.size(), values.values.data());
        cosa::hybrid.reference = network.forward(values);
        return cosa::hybrid.reference.values.data();
    } catch (const std::exception& error) {
        cosa::lastError = error.what();
        return nullptr;
    }
}

EMSCRIPTEN_KEEPALIVE uint32_t hybrid_reference_bytes() {
    return cosa::hybrid.reference.values.size() * sizeof(int32_t);
}

}
