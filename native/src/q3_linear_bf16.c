#include "cuda_kernel_loader.h"
#include <cuda_runtime_api.h>
#include <math.h>
#include <stdint.h>
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <pthread.h>
#endif

#ifdef _WIN32
static INIT_ONCE once = INIT_ONCE_STATIC_INIT;
#else
static pthread_once_t once = PTHREAD_ONCE_INIT;
#endif
static CUmodule module;
static CUfunction function;
static int init_status = EUHEDRAL_CUDA_KERNEL_UNAVAILABLE;
static void initialize(void) {
    init_status = euhedral_cuda_load_kernel((const void*)&once, "q3_linear_bf16.cu", "euhedral_q3_linear_bf16", &module, &function);
}
#ifdef _WIN32
static BOOL CALLBACK initialize_once(PINIT_ONCE state, PVOID parameter, PVOID* context) {
    (void)state; (void)parameter; (void)context;
    initialize();
    return TRUE;
}
#endif

int euhedral_cuda_linear_q3_bf16(const void* input, const void* weights, void* output,
        uint32_t rows, uint32_t in_features, uint32_t out_features, uint64_t weights_byte_size) {
    if (input == NULL || weights == NULL || output == NULL || rows == 0 || in_features == 0 || out_features == 0 || weights_byte_size == 0)
        return EUHEDRAL_CUDA_INVALID_ARGUMENT;
    int context_status = euhedral_cuda_bind_thread_context();
    if (context_status != EUHEDRAL_CUDA_SUCCESS) return context_status;
    uint64_t k_pad = ((uint64_t)in_features + 127u) / 128u * 128u;
    uint64_t groups = k_pad / 64u;
    uint64_t base = (uint64_t)out_features * groups * 24u;
    uint64_t scale_offset = (base + 255u) & ~255ull;
    uint64_t expected = scale_offset + (uint64_t)out_features * groups * 2u;
    if (expected != weights_byte_size) return EUHEDRAL_CUDA_FORMAT_MISMATCH;
    uint64_t grid = (uint64_t)rows * out_features;
    if (grid > 2147483647u) return EUHEDRAL_CUDA_SIZE_OVERFLOW;
#ifdef _WIN32
    if (!InitOnceExecuteOnce(&once, initialize_once, NULL, NULL)) return EUHEDRAL_CUDA_KERNEL_UNAVAILABLE;
#else
    if (pthread_once(&once, initialize) != 0) return EUHEDRAL_CUDA_KERNEL_UNAVAILABLE;
#endif
    if (init_status != EUHEDRAL_CUDA_SUCCESS) return init_status;
    CUdeviceptr input_ptr = (CUdeviceptr)(uintptr_t)input, weights_ptr = (CUdeviceptr)(uintptr_t)weights;
    CUdeviceptr output_ptr = (CUdeviceptr)(uintptr_t)output;
    unsigned int rows_arg = rows, in_arg = in_features, out_arg = out_features;
    unsigned long long scale_arg = scale_offset;
    void* params[] = {&input_ptr, &weights_ptr, &output_ptr, &rows_arg, &in_arg, &out_arg, &scale_arg};
    CUresult status = cuLaunchKernel(function, (unsigned int)grid, 1, 1, 128, 1, 1, 0, NULL, params, NULL);
    if (status != CUDA_SUCCESS) return (int)status;
    cudaError_t sync = cudaDeviceSynchronize();
    return sync == cudaSuccess ? EUHEDRAL_CUDA_SUCCESS : (int)sync;
}
