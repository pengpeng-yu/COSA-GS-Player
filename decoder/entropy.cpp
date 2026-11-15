#include "entropy.h"

#include <algorithm>
#include <limits>
#include <utility>

namespace cosa {
namespace {

constexpr uint32_t kPrecision = 16;
constexpr uint32_t kCdfTotal = 1u << kPrecision;
constexpr uint32_t kRansLowerBound = 1u << 23;
constexpr size_t kLookupStride = 257;

void validateCdf(const std::vector<uint16_t>& cdf) {
    // The initial zero and final 65536 are implicit. Empty intervals are valid.
    require(std::is_sorted(cdf.begin(), cdf.end()), "CDF boundaries are not sorted.");
}

// Byte-aligned rANS state updates adapted from ryg_rans (CC0).
// https://github.com/rygorous/ryg_rans
class RansDecoder {
public:
    explicit RansDecoder(Bytes stream) {
        Reader reader(stream);
        state_ = reader.u32();
        bytes_ = reader.take(reader.remaining());
        require(state_ >= kRansLowerBound && state_ < (kRansLowerBound << 8),
                "Invalid initial rANS state.");
    }

    uint32_t cumulative() const { return state_ & (kCdfTotal - 1); }

    void advance(uint32_t start, uint32_t frequency, uint32_t precision = kPrecision) {
        const uint32_t mask = (1u << precision) - 1;
        uint32_t value = frequency * (state_ >> precision) + (state_ & mask) - start;
        while (value < kRansLowerBound) {
            if (position_ == bytes_.size)
                require(false, "Truncated rANS stream.");
            value = (value << 8) | bytes_.data[position_++];
        }
        state_ = value;
    }

    uint32_t bit() {
        const uint32_t value = state_ & 1u;
        advance(value, 1, 1);
        return value;
    }

    uint32_t expGolomb() {
        uint32_t prefix = 0;
        while (bit() == 0) {
            if (++prefix > 31)
                require(false, "rANS escape code exceeds 32 bits.");
        }
        uint32_t quotient = 1u << prefix;
        for (uint32_t remaining = prefix; remaining > 0; --remaining)
            quotient |= bit() << (remaining - 1);
        return quotient - 1;
    }

    void finish() const {
        require(position_ == bytes_.size && state_ == kRansLowerBound,
                "rANS stream did not end at the expected state.");
    }

private:
    Bytes bytes_{};
    size_t position_ = 0;
    uint32_t state_ = 0;
};

class CategoricalTable {
public:
    explicit CategoricalTable(const std::vector<uint16_t>& cdf)
        : cdf_(cdf), inverse_(kCdfTotal) {
        require(cdf.size() <= static_cast<size_t>(std::numeric_limits<int16_t>::max()),
                "Categorical symbols exceed int16 range.");
        validateCdf(cdf);
        uint32_t start = 0;
        for (size_t symbol = 0; symbol <= cdf.size(); ++symbol) {
            const uint32_t end = symbol == cdf.size() ? kCdfTotal : cdf[symbol];
            std::fill(inverse_.begin() + start, inverse_.begin() + end,
                      static_cast<uint16_t>(symbol));
            start = end;
        }
    }

    int16_t decode(RansDecoder& decoder) const {
        const uint32_t symbol = inverse_[decoder.cumulative()];
        const uint32_t start = symbol == 0 ? 0 : cdf_[symbol - 1];
        const uint32_t end = symbol == cdf_.size() ? kCdfTotal : cdf_[symbol];
        decoder.advance(start, end - start);
        return static_cast<int16_t>(symbol);
    }

private:
    const std::vector<uint16_t>& cdf_;
    std::vector<uint16_t> inverse_;
};

int16_t decodeGaussianSymbol(
    RansDecoder& decoder, uint16_t index, const GaussianTables& tables) {
    if (index >= tables.cdfs.size())
        require(false, "Gaussian CDF index is out of range.");
    const auto& cdf = tables.cdfs[index];
    const auto* inverse = tables.inverseCdfs.data() + size_t(index) * kLookupStride;
    const uint32_t cumulative = decoder.cumulative();
    const uint32_t bucket = cumulative >> 8;
    const auto upper = std::upper_bound(
        cdf.begin() + inverse[bucket], cdf.begin() + inverse[bucket + 1], cumulative);
    const size_t symbol = static_cast<size_t>(upper - cdf.begin());
    const uint32_t start = symbol == 0 ? 0 : cdf[symbol - 1];
    const uint32_t end = symbol == cdf.size() ? kCdfTotal : cdf[symbol];
    decoder.advance(start, end - start);

    int64_t value = static_cast<int64_t>(symbol);
    if (symbol == cdf.size()) {
        // The backward encoder writes sign first, so the decoder reads it last.
        const uint32_t gamma = decoder.expGolomb() + 1;
        value = decoder.bit() ? -int64_t(gamma) : int64_t(gamma) + int64_t(cdf.size()) - 1;
    }
    value += tables.offsets[index];
    if (value < std::numeric_limits<int16_t>::min() ||
        value > std::numeric_limits<int16_t>::max())
        require(false, "Decoded Gaussian symbol exceeds int16 range.");
    return static_cast<int16_t>(value);
}

void checkCoordinateCount(size_t rows) {
    require(rows <= std::vector<int32_t>().max_size() / 3,
            "Octree coordinate array is too large.");
}

}  // namespace

GaussianTables GaussianTables::load(Bytes bytes) {
    Reader reader(bytes);
    const uint16_t count = reader.u16();
    require(count == 128, "Expected 128 Gaussian CDF tables.");
    GaussianTables tables;
    tables.cdfs.reserve(count);
    tables.offsets.reserve(count);
    tables.inverseCdfs.resize(size_t(count) * kLookupStride);
    for (size_t index = 0; index < count; ++index) {
        tables.offsets.push_back(reader.i32());
        const uint16_t length = reader.u16();
        tables.cdfs.push_back(reader.array<uint16_t>(length));
        const auto& cdf = tables.cdfs.back();
        validateCdf(cdf);
        auto* inverse = tables.inverseCdfs.data() + index * kLookupStride;
        size_t symbol = 0;
        for (size_t bucket = 0; bucket < kLookupStride; ++bucket) {
            const uint32_t cumulative = static_cast<uint32_t>(bucket) << 8;
            while (symbol < cdf.size() && cdf[symbol] <= cumulative)
                ++symbol;
            inverse[bucket] = static_cast<uint16_t>(symbol);
        }
    }
    require(reader.remaining() == 0, "Unexpected trailing Gaussian CDF data.");
    return tables;
}

void decodeGaussianInto(
    Bytes stream, const uint8_t* indices, size_t count,
    const GaussianTables& tables, std::vector<int16_t>& symbols) {
    require(tables.offsets.size() == tables.cdfs.size() &&
            tables.inverseCdfs.size() == tables.cdfs.size() * kLookupStride,
            "Gaussian CDF tables are incomplete.");
    Reader reader(stream);
    const uint32_t blocks = reader.u32();
    require(blocks > 0 && blocks <= reader.remaining() / sizeof(uint32_t),
            "Invalid Gaussian rANS block count.");
    const auto lengths = reader.array<uint32_t>(blocks);
    symbols.resize(count);
    for (uint32_t block = 0; block < blocks; ++block) {
        RansDecoder decoder(reader.take(lengths[block]));
        const size_t begin = uint64_t(count) * block / blocks;
        const size_t end = uint64_t(count) * (uint64_t(block) + 1) / blocks;
        for (size_t position = begin; position < end; ++position)
            symbols[position] = decodeGaussianSymbol(decoder, indices[position], tables);
        decoder.finish();
    }
    require(reader.remaining() == 0, "Unexpected trailing Gaussian rANS data.");
}

std::vector<int16_t> decodeMask(
    Bytes stream, const std::vector<uint16_t>& cdf, size_t count) {
    RansDecoder decoder(stream);
    CategoricalTable table(cdf);
    std::vector<int16_t> symbols(count);
    for (auto& symbol : symbols)
        symbol = table.decode(decoder);
    decoder.finish();
    return symbols;
}

IntMatrix decodeOctree(Bytes stream) {
    Reader reader(stream);
    const uint32_t levels = reader.u32();
    const uint32_t coarseCount = reader.u32();
    const uint16_t coarseCdfSize = reader.u16();
    require(levels <= 31, "Octree depth exceeds int32 coordinates.");
    std::vector<std::vector<uint16_t>> cdfs;
    cdfs.reserve(size_t(levels) + 1);
    cdfs.push_back(reader.array<uint16_t>(coarseCdfSize));
    for (uint32_t level = 0; level < levels; ++level)
        cdfs.push_back(reader.array<uint16_t>(255));

    // Coarse coordinates and all occupancy levels share one rANS state.
    RansDecoder decoder(reader.take(reader.remaining()));
    checkCoordinateCount(coarseCount);
    IntMatrix coordinates(coarseCount, 3);
    {
        CategoricalTable table(cdfs.front());
        for (auto& coordinate : coordinates.values)
            coordinate = table.decode(decoder);
    }
    for (uint32_t level = 0; level < levels; ++level) {
        CategoricalTable table(cdfs[level + 1]);
        std::vector<uint8_t> occupancies(coordinates.rows);
        size_t childCount = 0;
        for (auto& occupancy : occupancies) {
            occupancy = static_cast<uint8_t>(table.decode(decoder));
            for (uint32_t bits = occupancy; bits != 0; bits >>= 1)
                childCount += bits & 1u;
        }
        checkCoordinateCount(childCount);
        IntMatrix children(childCount, 3);
        size_t child = 0;
        for (size_t parent = 0; parent < coordinates.rows; ++parent) {
            const uint8_t occupancy = occupancies[parent];
            if (occupancy == 0)
                continue;
            const auto* coordinate = coordinates.values.data() + parent * 3;
            for (size_t axis = 0; axis < 3; ++axis) {
                if (coordinate[axis] > std::numeric_limits<int32_t>::max() / 2)
                    require(false, "Octree coordinate exceeds int32 range.");
            }
            // Parent order is unchanged. Occupied octants are visited 0 through 7.
            for (uint32_t octant = 0; octant < 8; ++octant) {
                if ((occupancy & (1u << octant)) == 0)
                    continue;
                for (size_t axis = 0; axis < 3; ++axis)
                    children.values[child * 3 + axis] =
                        coordinate[axis] * 2 + ((octant >> (2 - axis)) & 1u);
                ++child;
            }
        }
        coordinates = std::move(children);
    }
    decoder.finish();
    return coordinates;
}

}  // namespace cosa
