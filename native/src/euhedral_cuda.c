#include "euhedral_cuda.h"

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <stddef.h>
#include <stdint.h>

#if !defined(CUDA_VERSION) || CUDA_VERSION != 13010
#error "Euhedral CUDA ABI requires CUDA toolkit 13.1.x"
#endif

void* euhedral_cuda_malloc(uint64_t byte_size) {
    if (byte_size == 0 || byte_size > SIZE_MAX) {
        return NULL;
    }

    void* address = NULL;
    if (cudaMalloc(&address, (size_t) byte_size) != cudaSuccess) {
        return NULL;
    }
    return address;
}

int euhedral_cuda_free(void* address) {
    if (address == NULL) {
        return EUHEDRAL_CUDA_SUCCESS;
    }

    cudaError_t status = cudaFree(address);
    return status == cudaSuccess ? EUHEDRAL_CUDA_SUCCESS : (int) status;
}

int euhedral_cuda_device_memory_info(uint64_t* free_byte_size, uint64_t* total_byte_size) {
    if (free_byte_size == NULL || total_byte_size == NULL) {
        return EUHEDRAL_CUDA_INVALID_ARGUMENT;
    }

    size_t free_bytes = 0;
    size_t total_bytes = 0;
    cudaError_t status = cudaMemGetInfo(&free_bytes, &total_bytes);
    if (status != cudaSuccess) {
        return (int) status;
    }

    *free_byte_size = (uint64_t) free_bytes;
    *total_byte_size = (uint64_t) total_bytes;
    return EUHEDRAL_CUDA_SUCCESS;
}

int euhedral_cuda_copy_host_to_device(
        void* device_address,
        const void* host_address,
        uint64_t byte_size) {
    if (byte_size == 0) {
        return EUHEDRAL_CUDA_SUCCESS;
    }
    if (device_address == NULL || host_address == NULL) {
        return EUHEDRAL_CUDA_INVALID_ARGUMENT;
    }
    if (byte_size > SIZE_MAX) {
        return EUHEDRAL_CUDA_SIZE_OVERFLOW;
    }

    cudaError_t status = cudaMemcpy(
            device_address,
            host_address,
            (size_t) byte_size,
            cudaMemcpyHostToDevice);
    return status == cudaSuccess ? EUHEDRAL_CUDA_SUCCESS : (int) status;
}

int euhedral_cuda_copy_device_to_host(
        void* host_address,
        const void* device_address,
        uint64_t byte_size) {
    if (byte_size == 0) {
        return EUHEDRAL_CUDA_SUCCESS;
    }
    if (host_address == NULL || device_address == NULL) {
        return EUHEDRAL_CUDA_INVALID_ARGUMENT;
    }
    if (byte_size > SIZE_MAX) {
        return EUHEDRAL_CUDA_SIZE_OVERFLOW;
    }

    cudaError_t status = cudaMemcpy(
            host_address,
            device_address,
            (size_t) byte_size,
            cudaMemcpyDeviceToHost);
    return status == cudaSuccess ? EUHEDRAL_CUDA_SUCCESS : (int) status;
}
