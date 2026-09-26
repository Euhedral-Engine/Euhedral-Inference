#pragma once

namespace q3 {
static constexpr unsigned int kGroup = 64;
static constexpr unsigned int kGroupBytes = 24;

struct Layout {
    const unsigned char* codes;
    const unsigned short* scales;
    unsigned int groups;

    __device__ __forceinline__ Layout(
            const unsigned char* weights, unsigned int in_features, unsigned long long scale_offset)
            : codes(weights),
              scales((const unsigned short*)(weights + scale_offset)),
              groups(((in_features + 127u) / 128u) * 2u) {}

    // Linear group index for output row `out` and K group `k_group`.
    __device__ __forceinline__ unsigned long long group(unsigned int out, unsigned int k_group) const {
        return (unsigned long long)out * groups + k_group;
    }
    __device__ __forceinline__ const unsigned char* group_bytes(unsigned long long g) const {
        return codes + g * (unsigned long long)kGroupBytes;
    }
};


}  // namespace q3
