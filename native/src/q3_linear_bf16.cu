#include <cuda_bf16.h>
#include <mma.h>

// Q3_G64_FP16 linear kernels, built from small compile-time composable pieces.
//
// Layout of this file:
//   1. Numeric conversions (stateless, thread-local).
//   2. Layout: the packed Q3 storage contract shared by every path.
//   3. Primitives: packed-tile loading, activation loading, decode, scale
//      application, execution-tile staging, MMA consumption, FP32 accumulation,
//      and output writeback.
//   4. Strategies: schedules that assemble primitives for one execution path.
//   5. extern "C" entry points: the stable symbols loaded by q3_linear_bf16.c.
//
// Ownership conventions (every primitive states its own terms against these):
//   - OWNS:     registers the primitive creates and returns by value.
//   - BORROWS:  memory it reads or writes only for the duration of the call; it
//               never retains a pointer after returning.
//   - PRODUCES: the value or buffer contents that become valid when it returns.
//   - REUSE:    when the caller may overwrite a borrowed input or read an output.
//   - SCOPE:    thread-local, warp-collective (full-mask shuffles: all 32 lanes
//               must call with warp-uniform control flow), or CTA-cooperative
//               (all CTA threads call; work is strided, no barrier inside).
//
// Strategies alone own shared-memory storage, barriers, loop order, and tile
// geometry. Except for the CTA reduction used by the scalar reference, no
// primitive calls __syncthreads(): a strategy decides when a staged tile becomes
// visible to its consumers and when it may be overwritten. This keeps future
// schedules (double buffering, producer/consumer warps, larger M/N tiles,
// split-K partials) free to reorder the same pieces without rewriting them.
//
// Nothing here uses virtual dispatch or runtime selection. Geometry is template
// parameters and everything is force-inlined, so each entry point compiles to a
// single straight kernel.

namespace q3 {

// ---------------------------------------------------------------------------
// 1. Numeric conversions
// ---------------------------------------------------------------------------

static __device__ __forceinline__ float bf16_to_float(unsigned short value) {
    return __uint_as_float((unsigned int)value << 16);
}
static __device__ __forceinline__ float fp16_to_float(unsigned short value) {
    unsigned int sign = ((unsigned int)value & 0x8000u) << 16;
    unsigned int exponent = ((unsigned int)value >> 10) & 31u;
    unsigned int mantissa = (unsigned int)value & 1023u;
    if (exponent == 0u) {
        if (mantissa == 0u) return __uint_as_float(sign);
        int shift = 0;
        while ((mantissa & 1024u) == 0u) { mantissa <<= 1; --shift; }
        mantissa &= 1023u;
        return __uint_as_float(sign | ((unsigned int)(113 + shift) << 23) | (mantissa << 13));
    }
    if (exponent == 31u) return __uint_as_float(sign | 0x7f800000u | (mantissa << 13));
    return __uint_as_float(sign | ((exponent + 112u) << 23) | (mantissa << 13));
}
static __device__ __forceinline__ unsigned short float_to_bf16(float value) {
    unsigned int bits = __float_as_uint(value);
    bits += 0x7fffu + ((bits >> 16) & 1u);
    return (unsigned short)(bits >> 16);
}

// ---------------------------------------------------------------------------
// 2. Packed Q3 storage contract (ROW_SPLIT_K128_V1, Q3_G64_FP16)
// ---------------------------------------------------------------------------

// Each output row stores ceil(K / 128) * 2 groups of 64 signed 3-bit codes
// (24 bytes per group, little-endian bit stream), followed after a 256-byte
// aligned offset by one FP16 scale per group. Layout is a read-only view over the
// caller's weight allocation; it owns no memory.
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

// ---------------------------------------------------------------------------
// 3. Primitives
// ---------------------------------------------------------------------------

// --- Packed Q3 tile loading -------------------------------------------------

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

// --- Q3 decode / dequantization --------------------------------------------

// Decode one signed 3-bit code (range [-4, 3]) from slot 0 or 1 of a 6-bit field.
// SCOPE: thread-local; pure.
static __device__ __forceinline__ int decode_code(unsigned int pair, unsigned int slot) {
    int code = (pair >> (slot * 3)) & 7;
    code -= (code & 4) ? 8 : 0;
    return code;
}

// --- Scale application ------------------------------------------------------
//
// Two orders exist and are separate numerical contracts; do not interchange them.

// Dequantized weight value code * scale. Exact in FP32 (at most 14 significant
// bits: FP16 mantissa times a 3-bit code). Used when a weight is staged for MMA.
static __device__ __forceinline__ float apply_scale(int code, float scale) {
    return (float)code * scale;
}

// Activation contribution in the scalar reference's FP32 order: (x * code) * scale.
// SIMT paths use this so their per-stripe sums match the reference bit for bit.
static __device__ __forceinline__ float scaled_product(float x, int code, float scale) {
    return x * (float)code * scale;
}

// --- Activation tile loading -----------------------------------------------

// Stage a ROWS x K_TILE activation tile from global BF16 into a borrowed shared
// tile (row-major, leading dimension K_TILE), zero-filling rows >= `rows` and
// columns >= `in_features`.
// SCOPE: CTA-cooperative over THREADS threads; no barrier. The strategy must
// barrier before any consumer reads `tile`, and again before it is re-staged.
template<int ROWS, int K_TILE, int THREADS>
static __device__ __forceinline__ void stage_activation_tile(
        __nv_bfloat16* tile, const unsigned short* input, unsigned int rows, unsigned int in_features,
        unsigned int row_start, unsigned int k_base, unsigned int thread) {
    for (unsigned int i = thread; i < ROWS * K_TILE; i += THREADS) {
        unsigned int r = row_start + i / K_TILE, k = k_base + i % K_TILE;
        tile[i] = __float2bfloat16(r < rows && k < in_features
                ? bf16_to_float(input[(unsigned long long)r * in_features + k]) : 0.0f);
    }
}

// Load a lane's K-stripe activations into registers for R rows: x[r][s] holds
// column k_base + lane + 32*s, zero outside the matrix.
// SCOPE: thread-local reads; OWNS `x` (caller registers) until the next K step.
template<int R, int STRIPES>
static __device__ __forceinline__ void load_activation_stripes(
        float (&x)[R][STRIPES], const unsigned short* input, unsigned int rows, unsigned int in_features,
        unsigned int first_row, unsigned int k_base, unsigned int lane) {
    #pragma unroll
    for (int r = 0; r < R; r++) {
        #pragma unroll
        for (int s = 0; s < STRIPES; s++) {
            unsigned int column = k_base + lane + s * 32;
            unsigned long long offset = (unsigned long long)(first_row + r) * in_features + column;
            x[r][s] = first_row + r < rows && column < in_features ? bf16_to_float(input[offset]) : 0;
        }
    }
}

// --- Execution-tile staging -------------------------------------------------

// Store an FP32 weight as two BF16 components (hi + lo) in borrowed shared tiles.
// hi = rn(weight), lo = rn(weight - hi). For Q3 weights this pair is exact, so the
// MMA path does not silently round the FP16-scaled value to BF16.
// SCOPE: thread-local; the element index is owned by the calling thread.
static __device__ __forceinline__ void stage_split_weight(
        __nv_bfloat16* hi, __nv_bfloat16* lo, unsigned int i, float weight) {
    hi[i] = __float2bfloat16(weight);
    lo[i] = __float2bfloat16(weight - __bfloat162float(hi[i]));
}

// Decode and stage one K group for COLS output columns into borrowed hi/lo tiles
// (column-major: column c occupies [c * kGroup, (c + 1) * kGroup)). Each warp
// stages columns warp, warp + WARPS, ...; columns >= out_features stage zeros.
// SCOPE: each warp is warp-collective over its columns; no barrier. The strategy
// must barrier before MMA consumption and again before re-staging.
template<int COLS, int WARPS>
static __device__ __forceinline__ void stage_weight_tile(
        __nv_bfloat16* hi, __nv_bfloat16* lo, const Layout& w, unsigned int out_start,
        unsigned int out_features, unsigned int k_base, unsigned int warp, unsigned int lane) {
    for (unsigned int col = warp; col < COLS; col += WARPS) {
        unsigned int codes = 0;
        float scale = 0.0f;
        if (out_start + col < out_features) {
            unsigned long long g = w.group(out_start + col, k_base / kGroup);
            codes = load_packed_pair(w, g, lane);
            scale = load_group_scale(w, g, lane);
        }
        #pragma unroll
        for (int p = 0; p < 2; p++)
            stage_split_weight(hi, lo, col * kGroup + lane * 2 + p, apply_scale(decode_code(codes, p), scale));
    }
}

// --- WMMA warp geometry, MMA consumption, and FP32 accumulation -------------

namespace wmma = nvcuda::wmma;
using AccumulatorFragment = wmma::fragment<wmma::accumulator, 16, 16, 16, float>;

// Warp placement inside a CTA output tile. WARPS_M x WARPS_N warps each own
// M_FRAGS vertically strided 16x16 fragments in a single 16-column band. The CTA
// tile is (16 * WARPS_M * M_FRAGS) x (16 * WARPS_N).
template<int WARPS_M, int WARPS_N, int M_FRAGS>
struct WarpTile {
    static constexpr int kWarps = WARPS_M * WARPS_N;
    static constexpr int kRows = 16 * WARPS_M * M_FRAGS;
    static constexpr int kCols = 16 * WARPS_N;
    static constexpr int kFrags = M_FRAGS;
    static __device__ __forceinline__ unsigned int row(unsigned int warp, int m) {
        return (warp / WARPS_N) * 16 + m * 16 * WARPS_M;
    }
    static __device__ __forceinline__ unsigned int col(unsigned int warp) { return (warp % WARPS_N) * 16; }
};

// FP32 accumulators owned by one warp for the whole K loop. Only fill() and
// consume_mma_tile() modify them; they become final after the last K step.
template<int M_FRAGS>
struct Accumulators {
    AccumulatorFragment frag[M_FRAGS];
    __device__ __forceinline__ void fill() {
        #pragma unroll
        for (int m = 0; m < M_FRAGS; m++) wmma::fill_fragment(frag[m], 0.0f);
    }
};

// Consume one staged K tile: acc += A * (B_hi + B_lo) for this warp's fragments.
// SCOPE: warp-collective (WMMA). BORROWS: the shared A and B tiles, read-only; the
// strategy must not re-stage them until every warp has returned and barriered.
// Each B fragment is loaded once and reused across all M_FRAGS accumulators.
template<class Tile, int K_TILE, int LDA, int LDB>
static __device__ __forceinline__ void consume_mma_tile(
        Accumulators<Tile::kFrags>& acc, const __nv_bfloat16* a, const __nv_bfloat16* b_hi,
        const __nv_bfloat16* b_lo, unsigned int warp) {
    #pragma unroll
    for (unsigned int k = 0; k < K_TILE; k += 16) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> af[Tile::kFrags];
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> bf;
        #pragma unroll
        for (int m = 0; m < Tile::kFrags; m++) wmma::load_matrix_sync(af[m], a + Tile::row(warp, m) * LDA + k, LDA);
        wmma::load_matrix_sync(bf, b_hi + Tile::col(warp) * LDB + k, LDB);
        #pragma unroll
        for (int m = 0; m < Tile::kFrags; m++) wmma::mma_sync(acc.frag[m], af[m], bf, acc.frag[m]);
        wmma::load_matrix_sync(bf, b_lo + Tile::col(warp) * LDB + k, LDB);
        #pragma unroll
        for (int m = 0; m < Tile::kFrags; m++) wmma::mma_sync(acc.frag[m], af[m], bf, acc.frag[m]);
    }
}

// Combine a lane's four K-stripe FP32 sums, then reduce across the warp. The
// pairing ((s0 + s2) + (s1 + s3)) and shuffle tree match the scalar reference.
// SCOPE: warp-collective. PRODUCES: the full sum in lane 0.
static __device__ __forceinline__ float reduce_stripes(const float (&s)[4]) {
    float sum = (s[0] + s[2]) + (s[1] + s[3]);
    #pragma unroll
    for (int distance = 16; distance; distance >>= 1) sum += __shfl_down_sync(0xffffffffu, sum, distance);
    return sum;
}

// CTA tree reduction of one FP32 value per thread for 128 threads.
// SCOPE: CTA-collective, including its barriers. BORROWS: `partial[128]` until
// return. PRODUCES: the total in partial[0], visible to every thread.
static __device__ __forceinline__ void reduce_cta_128(float* partial, float value) {
    partial[threadIdx.x] = value;
    __syncthreads();
    for (unsigned int stride = 64; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
}

// --- Output writeback -------------------------------------------------------

// Write one FP32 result as BF16 (round-to-nearest-even). Bounds are the caller's.
static __device__ __forceinline__ void write_bf16(
        unsigned short* output, unsigned int row, unsigned int col, unsigned int out_features, float value) {
    output[(unsigned long long)row * out_features + col] = float_to_bf16(value);
}

// Store this warp's accumulators to a borrowed shared FP32 tile (row-major,
// leading dimension Tile::kCols). SCOPE: warp-collective; no barrier.
template<class Tile>
static __device__ __forceinline__ void store_accumulators(
        float* result, Accumulators<Tile::kFrags>& acc, unsigned int warp) {
    #pragma unroll
    for (int m = 0; m < Tile::kFrags; m++)
        wmma::store_matrix_sync(result + Tile::row(warp, m) * Tile::kCols + Tile::col(warp),
                acc.frag[m], Tile::kCols, wmma::mem_row_major);
}

// Write a ROWS x COLS FP32 shared tile to global BF16, skipping out-of-matrix
// elements. SCOPE: CTA-cooperative; the strategy barriers after the stores above.
template<int ROWS, int COLS, int THREADS>
static __device__ __forceinline__ void write_output_tile(
        unsigned short* output, const float* result, unsigned int rows, unsigned int out_features,
        unsigned int row_start, unsigned int out_start, unsigned int thread) {
    for (unsigned int i = thread; i < ROWS * COLS; i += THREADS) {
        unsigned int r = row_start + i / COLS, n = out_start + i % COLS;
        if (r < rows && n < out_features) write_bf16(output, r, n, out_features, result[i]);
    }
}

// ---------------------------------------------------------------------------
// 4. Strategies
// ---------------------------------------------------------------------------

// Scalar reference: one CTA per (row, output); threads stride K. Kept independent
// of every shared-load primitive so it remains an oracle for the tiled paths.
static __device__ __forceinline__ void scalar_reference(
        const unsigned short* input, const unsigned char* weights, unsigned short* output,
        unsigned int rows, unsigned int in_features, unsigned int out_features,
        unsigned long long scale_offset, float* partial) {
    unsigned int linear = blockIdx.x;
    unsigned int row = linear / out_features, out = linear % out_features;
    if (row >= rows) return;
    const Layout w(weights, in_features, scale_offset);
    float sum = 0.0f;
    for (unsigned int k = threadIdx.x; k < in_features; k += blockDim.x) {
        unsigned long long g = w.group(out, k / kGroup);
        int code = load_code_at(w.group_bytes(g), k % kGroup);
        // The reference has always formed its scale index in 32-bit arithmetic, which
        // wraps once a single tensor reaches 2^32 groups (>= 104 GiB). Preserved as-is;
        // the tiled paths use the 64-bit group index.
        unsigned int scale_index = out * w.groups + k / kGroup;
        sum = sum + scaled_product(bf16_to_float(input[(unsigned long long)row * in_features + k]), code,
                load_scale_at(w, scale_index));
    }
    reduce_cta_128(partial, sum);
    if (threadIdx.x == 0) write_bf16(output, row, out, out_features, partial[0]);
}

// Cooperative decode for R <= 4 token rows. Each of 4 warps owns 2 output
// columns; a CTA covers 8 columns x R rows. Per 128-wide K step a lane holds four
// K-stripe activations per row and accumulates four FP32 stripes per (row, col),
// reusing each loaded activation across both columns.
template<int R>
static __device__ __forceinline__ void cooperative_decode(
        const unsigned short* input, const unsigned char* weights, unsigned short* output,
        unsigned int rows, unsigned int in_features, unsigned int out_features,
        unsigned long long scale_offset) {
    constexpr int kWarpCols = 2, kStripes = 4;
    const unsigned int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const unsigned int output_tiles = (out_features + 7u) / 8u;
    const unsigned int first_out = (blockIdx.x % output_tiles) * 8 + warp * kWarpCols;
    const unsigned int first_row = (blockIdx.x / output_tiles) * R;
    const Layout w(weights, in_features, scale_offset);
    float sums[R][kWarpCols][kStripes] = {};
    for (unsigned int k = 0; k < in_features; k += 2 * kGroup) {
        float x[R][kStripes];
        load_activation_stripes<R, kStripes>(x, input, rows, in_features, first_row, k, lane);
        #pragma unroll
        for (int n = 0; n < kWarpCols; n++) {
            if (first_out + n < out_features) {
                #pragma unroll
                for (int half = 0; half < 2; half++) {
                    unsigned long long g = w.group(first_out + n, k / kGroup + half);
                    unsigned int pairs = load_packed_pair(w, g, lane);
                    float scale = load_group_scale(w, g, lane);
                    #pragma unroll
                    for (int p = 0; p < 2; p++) {
                        int code = decode_code(stripe_pair(pairs, lane, p), lane & 1);
                        #pragma unroll
                        for (int r = 0; r < R; r++)
                            sums[r][n][half * 2 + p] += scaled_product(x[r][half * 2 + p], code, scale);
                    }
                }
            }
        }
    }
    #pragma unroll
    for (int r = 0; r < R; r++) {
        #pragma unroll
        for (int n = 0; n < kWarpCols; n++) {
            float sum = reduce_stripes(sums[r][n]);
            if (lane == 0 && first_row + r < rows && first_out + n < out_features)
                write_bf16(output, first_row + r, first_out + n, out_features, sum);
        }
    }
}

// Tiled WMMA prefill. The CTA owns Tile::kRows x Tile::kCols outputs and consumes
// K one G64 group at a time:
//   stage A (activations) + stage B (decoded hi/lo weights)  -> barrier
//   every warp consumes the staged tile into FP32 accumulators -> barrier
// then stores accumulators to shared, barriers, and writes BF16 output. The two
// per-step barriers are the single-buffer reuse edges: staged tiles are visible
// only after the first, and may be overwritten only after the second.
template<class Tile>
static __device__ __forceinline__ void tiled_prefill(
        const unsigned short* input, const unsigned char* weights, unsigned short* output,
        unsigned int rows, unsigned int in_features, unsigned int out_features,
        unsigned long long scale_offset, __nv_bfloat16* a, __nv_bfloat16* b_hi, __nv_bfloat16* b_lo,
        float* result) {
    constexpr int kThreads = Tile::kWarps * 32;
    const unsigned int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const unsigned int output_tiles = (out_features + Tile::kCols - 1u) / Tile::kCols;
    const unsigned int row_start = (blockIdx.x / output_tiles) * Tile::kRows;
    const unsigned int out_start = (blockIdx.x % output_tiles) * Tile::kCols;
    const Layout w(weights, in_features, scale_offset);
    Accumulators<Tile::kFrags> acc;
    acc.fill();
    for (unsigned int base = 0; base < in_features; base += kGroup) {
        stage_activation_tile<Tile::kRows, kGroup, kThreads>(
                a, input, rows, in_features, row_start, base, threadIdx.x);
        stage_weight_tile<Tile::kCols, Tile::kWarps>(b_hi, b_lo, w, out_start, out_features, base, warp, lane);
        __syncthreads();
        consume_mma_tile<Tile, kGroup, kGroup, kGroup>(acc, a, b_hi, b_lo, warp);
        __syncthreads();
    }
    store_accumulators<Tile>(result, acc, warp);
    __syncthreads();
    write_output_tile<Tile::kRows, Tile::kCols, kThreads>(
            output, result, rows, out_features, row_start, out_start, threadIdx.x);
}

// Tile geometries used by the current routes. 2 x 2 warps with one or two
// fragments per warp give 32 x 32 and 64 x 32 CTA tiles. Host dispatch in
// q3_linear_bf16.c sizes grids with the same constants.
using Prefill32 = WarpTile<2, 2, 1>;
using Prefill64 = WarpTile<2, 2, 2>;

}  // namespace q3

// ---------------------------------------------------------------------------
// 5. Entry points (stable symbols; each owns its shared storage)
// ---------------------------------------------------------------------------

extern "C" __global__ __launch_bounds__(128) void euhedral_q3_linear_bf16(
        const unsigned short* input, const unsigned char* weights, unsigned short* output,
        unsigned int rows, unsigned int in_features, unsigned int out_features,
        unsigned long long scale_offset) {
    __shared__ float partial[128];
    q3::scalar_reference(input, weights, output, rows, in_features, out_features, scale_offset, partial);
}

#define EUHEDRAL_Q3_DECODE_KERNEL(R) \
extern "C" __global__ __launch_bounds__(128) void euhedral_q3_decode_##R( \
        const unsigned short* input, const unsigned char* weights, unsigned short* output, \
        unsigned int rows, unsigned int in_features, unsigned int out_features, unsigned long long scale_offset) { \
    q3::cooperative_decode<R>(input, weights, output, rows, in_features, out_features, scale_offset); \
}
EUHEDRAL_Q3_DECODE_KERNEL(1)
EUHEDRAL_Q3_DECODE_KERNEL(2)
EUHEDRAL_Q3_DECODE_KERNEL(4)
#undef EUHEDRAL_Q3_DECODE_KERNEL

// 32 token rows x 32 outputs; general Q3 prefill route.
extern "C" __global__ __launch_bounds__(128) void euhedral_q3_prefill(
        const unsigned short* input, const unsigned char* weights, unsigned short* output,
        unsigned int rows, unsigned int in_features, unsigned int out_features,
        unsigned long long scale_offset) {
    using Tile = q3::Prefill32;
    __shared__ __align__(32) __nv_bfloat16 a[Tile::kRows * q3::kGroup];
    __shared__ __align__(32) __nv_bfloat16 b_hi[Tile::kCols * q3::kGroup];
    __shared__ __align__(32) __nv_bfloat16 b_lo[Tile::kCols * q3::kGroup];
    __shared__ __align__(32) float result[Tile::kRows * Tile::kCols];
    q3::tiled_prefill<Tile>(input, weights, output, rows, in_features, out_features, scale_offset,
            a, b_hi, b_lo, result);
}

// 64 token rows x 32 outputs; shape-gated route for the large MLP projections.
// Reuses each decoded weight tile across twice as many rows. Optional symbol:
// host dispatch falls back to euhedral_q3_prefill when it is absent.
extern "C" __global__ __launch_bounds__(128) void euhedral_q3_prefill_64(
        const unsigned short* input, const unsigned char* weights, unsigned short* output,
        unsigned int rows, unsigned int in_features, unsigned int out_features,
        unsigned long long scale_offset) {
    using Tile = q3::Prefill64;
    __shared__ __align__(32) __nv_bfloat16 a[Tile::kRows * q3::kGroup];
    __shared__ __align__(32) __nv_bfloat16 b_hi[Tile::kCols * q3::kGroup];
    __shared__ __align__(32) __nv_bfloat16 b_lo[Tile::kCols * q3::kGroup];
    __shared__ __align__(32) float result[Tile::kRows * Tile::kCols];
    q3::tiled_prefill<Tile>(input, weights, output, rows, in_features, out_features, scale_offset,
            a, b_hi, b_lo, result);
}
