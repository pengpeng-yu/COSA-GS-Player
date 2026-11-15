#pragma once

#include "common.h"

#include <cstdint>
#include <vector>

namespace cosa {

struct GaussianTables {
    std::vector<std::vector<uint16_t>> cdfs;
    std::vector<int32_t> offsets;
    std::vector<uint16_t> inverseCdfs;

    static GaussianTables load(Bytes bytes);
};

void decodeGaussianInto(
    Bytes stream, const uint8_t* indices, size_t count,
    const GaussianTables& tables, std::vector<int16_t>& symbols);

std::vector<int16_t> decodeMask(
    Bytes stream, const std::vector<uint16_t>& cdf, size_t count);

IntMatrix decodeOctree(Bytes stream);

}  // namespace cosa
