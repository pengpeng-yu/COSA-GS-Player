#pragma once

#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace cosa {

constexpr int32_t FIXED_ONE = 1 << 20;

inline void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

inline void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

struct Bytes {
    const uint8_t* data;
    size_t size;
};

// The codec and WebAssembly use little-endian storage. memcpy also supports
// unaligned serialized parameters without aliasing or alignment assumptions.
class Reader {
    Bytes bytes_;
    size_t position_ = 0;

    template<class T>
    T scalar() {
        const auto bytes = take(sizeof(T));
        T value;
        std::memcpy(&value, bytes.data, sizeof(T));
        return value;
    }

public:
    explicit Reader(Bytes bytes) : bytes_(bytes) {}
    size_t remaining() const { return bytes_.size - position_; }

    Bytes take(size_t count) {
        require(count <= remaining(), "Truncated codec stream");
        const Bytes result{bytes_.data + position_, count};
        position_ += count;
        return result;
    }

    uint8_t u8() { return scalar<uint8_t>(); }
    uint16_t u16() { return scalar<uint16_t>(); }
    uint32_t u32() { return scalar<uint32_t>(); }
    uint64_t u64() { return scalar<uint64_t>(); }
    int32_t i32() { return scalar<int32_t>(); }
    float f32() { return scalar<float>(); }
    double f64() { return scalar<double>(); }

    template<class T>
    std::vector<T> array(size_t count) {
        static_assert(std::is_trivially_copyable_v<T>);
        require(count <= remaining() / sizeof(T), "Truncated codec array");
        const auto bytes = take(count * sizeof(T));
        std::vector<T> result(count);
        if (count) std::memcpy(result.data(), bytes.data, bytes.size);
        return result;
    }
};

struct IntMatrix {
    size_t rows = 0;
    size_t cols = 0;
    std::vector<int32_t> values;

    IntMatrix() = default;
    IntMatrix(size_t rowCount, size_t columnCount) : rows(rowCount), cols(columnCount) {
        require(!cols || rows <= std::numeric_limits<size_t>::max() / cols,
                "Codec matrix size overflow");
        values.resize(rows * cols);
    }
};

} // namespace cosa
