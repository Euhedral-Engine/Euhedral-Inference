#ifndef EUHEDRAL_CUDA_KERNEL_LOADER_H
#define EUHEDRAL_CUDA_KERNEL_LOADER_H

#include "euhedral_cuda.h"
#include <cuda.h>

int euhedral_cuda_bind_thread_context(void);

int euhedral_cuda_load_kernel(
        const void* anchor,
        const char* source_name,
        const char* function_name,
        CUmodule* module,
        CUfunction* function);

#endif
