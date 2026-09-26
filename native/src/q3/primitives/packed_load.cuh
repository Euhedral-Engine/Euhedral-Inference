#pragma once
#include "../layout.cuh"
#include "../numeric.cuh"

namespace q3 {
// Warp-cooperative load of one packed G64 group.
// SCOPE: warp-collective. BORROWS: the group's 24 global bytes for this call.
// PRODUCES (OWNS): a per-lane 6-bit field holding the two codes for K offsets
// 2*lane and 2*lane+1 within the group. Lanes 0..5 each issue one 32-bit read;
// shuffles hand every lane the bits it needs, so no lane reloads a word.
static __device__ __forceinline__ unsigned int load_packed_pair(
        const Layout& w, unsigned long long g, unsigned int lane) {
    unsigned int word = lane < 6 ? ((const unsigned int*)w.group_bytes(g))[lane] : 0;
    unsigned int bit = lane * 6u;
    unsigned int lo = __shfl_sync(0xffffffffu, word, bit >> 5);
    unsigned int hi = __shfl_sync(0xffffffffu, word, (bit >> 5) + 1);
    return (unsigned int)((((unsigned long long)hi << 32) | lo) >> (bit & 31u)) & 63u;
}

// Reference unpack of one code directly from packed bytes.
// SCOPE: thread-local. BORROWS: two global bytes. PRODUCES: the signed code.
// Used by the scalar reference, which deliberately shares no loads.
// Always reads byte (index * 3) / 8 + 1, so code 63 reads byte 24: one past its
// group. In the packed layout that byte is the next group, alignment padding
// before the scales, or the first scale byte; a standalone group buffer must
// carry one byte of tail padding.
static __device__ __forceinline__ int load_code_at(const unsigned char* group_bytes, unsigned int index) {
    unsigned int bit = index * 3u;
    unsigned int word = (unsigned int)group_bytes[bit >> 3];
    word |= (unsigned int)group_bytes[(bit >> 3) + 1u] << 8;
    return (int)((word >> (bit & 7u)) & 7u) - (((word >> (bit & 7u)) & 7u) >= 4u ? 8 : 0);
}

// Warp-cooperative load of one group's scale.
// SCOPE: warp-collective. Lane 0 reads the FP16 scale once and broadcasts it.
// PRODUCES: the FP32 scale, identical in every lane.
static __device__ __forceinline__ float load_group_scale(const Layout& w, unsigned long long g, unsigned int lane) {
    return fp16_to_float(__shfl_sync(0xffffffffu, lane == 0 ? (unsigned int)w.scales[g] : 0u, 0));
}

// Thread-local scale load for the scalar reference.
static __device__ __forceinline__ float load_scale_at(const Layout& w, unsigned long long g) {
    return fp16_to_float(w.scales[g]);
}

// Move a packed-pair field from load ownership (lane owns K = 2*lane, 2*lane+1)
// to K-stripe ownership (lane owns K = lane + 32*stripe within the group).
// SCOPE: warp-collective. BORROWS: every lane's `pairs` register for this call.
// PRODUCES: the 6-bit field containing this lane's stripe code; the code is in
// slot (lane & 1). `pairs` may be discarded after the last stripe is taken.
static __device__ __forceinline__ unsigned int stripe_pair(unsigned int pairs, unsigned int lane, int stripe) {
    return __shfl_sync(0xffffffffu, pairs, lane / 2 + stripe * 16);
}


}  // namespace q3
