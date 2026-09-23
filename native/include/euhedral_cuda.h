#ifndef EUHEDRAL_CUDA_H
#define EUHEDRAL_CUDA_H

#include <stdint.h>

#ifdef _WIN32
#define EUHEDRAL_CUDA_EXPORT __declspec(dllexport)
#else
#define EUHEDRAL_CUDA_EXPORT __attribute__((visibility("default")))
#endif

#define EUHEDRAL_CUDA_SUCCESS 0
#define EUHEDRAL_CUDA_INVALID_ARGUMENT (-1)
#define EUHEDRAL_CUDA_SIZE_OVERFLOW (-2)
#define EUHEDRAL_CUDA_FORMAT_MISMATCH (-3)
#define EUHEDRAL_CUDA_KERNEL_UNAVAILABLE (-4)

#ifdef __cplusplus
extern "C" {
#endif

EUHEDRAL_CUDA_EXPORT void* euhedral_cuda_malloc(uint64_t byte_size);
EUHEDRAL_CUDA_EXPORT int euhedral_cuda_free(void* address);
EUHEDRAL_CUDA_EXPORT int euhedral_cuda_device_memory_info(uint64_t* free_byte_size, uint64_t* total_byte_size);

EUHEDRAL_CUDA_EXPORT int euhedral_cuda_copy_host_to_device(
        void* device_address,
        const void* host_address,
        uint64_t byte_size);

EUHEDRAL_CUDA_EXPORT int euhedral_cuda_copy_device_to_host(
        void* host_address,
        const void* device_address,
        uint64_t byte_size);

EUHEDRAL_CUDA_EXPORT int euhedral_cuda_embed_q3(
        const int32_t* device_token_ids,
        const void* device_embedding_weights,
        void* device_hidden_state,
        uint32_t token_count,
        uint32_t vocabulary_size,
        uint32_t hidden_size,
        uint64_t embedding_byte_size);

EUHEDRAL_CUDA_EXPORT int euhedral_cuda_rms_norm_bf16(
        const void* device_input,
        const void* device_weight,
        void* device_output,
        uint32_t rows,
        uint32_t width,
        float epsilon);

EUHEDRAL_CUDA_EXPORT int euhedral_cuda_linear_q3_bf16(
        const void* device_input,
        const void* device_weights,
        void* device_output,
        uint32_t rows,
        uint32_t in_features,
        uint32_t out_features,
        uint64_t weights_byte_size);

EUHEDRAL_CUDA_EXPORT int euhedral_cuda_synchronize(void);

#ifdef __cplusplus
}
#endif

#endif
