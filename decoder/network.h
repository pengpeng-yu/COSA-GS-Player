#pragma once

#include "common.h"

namespace cosa {

// Exact arithmetic used when normalizing decoded coordinates.
// Signed half ties are rounded away from zero.
int64_t roundDivide(int64_t value, int64_t positiveDenominator);
int32_t checkedI32(int64_t value);

class IntegerNetwork {
public:
    // All intermediate Linear layers have GELU. finalGelu describes the last.
    static IntegerNetwork read(Reader& reader, const std::vector<int>& widths,
                               bool finalGelu = false);
    IntMatrix forward(const IntMatrix& input) const;
    size_t inputWidth() const { return layers_.front().inputWidth; }
    size_t outputWidth() const { return layers_.back().outputWidth; }

    // GPU layout uses an 8-word header followed by each layer's 8-word header,
    // packed INT8 weights, INT32 biases and UINT32 multipliers.
    std::vector<uint32_t> gpuParameters() const;

private:
    struct Requant {
        uint32_t multiplier = 1;
        int32_t shift = 0;
        int32_t zeroPoint = 0;
    };

    struct Layer {
        size_t inputWidth = 0;
        size_t outputWidth = 0;
        std::vector<int8_t> weights;
        std::vector<int32_t> bias;
        std::vector<uint32_t> multipliers;
        int32_t shift = 0;
        Requant output;
        bool gelu = false;
    };

    Requant input_;
    std::vector<Layer> layers_;
};

} // namespace cosa
