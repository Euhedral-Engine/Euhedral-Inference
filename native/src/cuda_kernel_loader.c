#ifndef _WIN32
#define _GNU_SOURCE
#endif
#include "cuda_kernel_loader.h"
#include <cuda_runtime_api.h>
#include <nvrtc.h>
#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#ifndef PATH_MAX
#define PATH_MAX 4096
#endif

int euhedral_cuda_bind_thread_context(void) {
    int device = 0;
    cudaError_t status = cudaGetDevice(&device);
    if (status == cudaSuccess) status = cudaSetDevice(device);
    return status == cudaSuccess ? EUHEDRAL_CUDA_SUCCESS : (int)status;
}

int euhedral_cuda_load_kernel(const void* anchor, const char* source_name, const char* function_name,
        CUmodule* module, CUfunction* function) {
    char library_path[PATH_MAX];
#ifdef _WIN32
    HMODULE owner = NULL;
    DWORD flags = GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT;
    if (!GetModuleHandleExA(flags, (LPCSTR)anchor, &owner)) return EUHEDRAL_CUDA_KERNEL_UNAVAILABLE;
    DWORD copied = GetModuleFileNameA(owner, library_path, sizeof(library_path));
    if (copied == 0 || copied >= sizeof(library_path)) return EUHEDRAL_CUDA_KERNEL_UNAVAILABLE;
#else
    Dl_info info;
    if (dladdr(anchor, &info) == 0 || info.dli_fname == NULL) return EUHEDRAL_CUDA_KERNEL_UNAVAILABLE;
    size_t copied = strlen(info.dli_fname);
    if (copied >= sizeof(library_path)) return EUHEDRAL_CUDA_KERNEL_UNAVAILABLE;
    memcpy(library_path, info.dli_fname, copied + 1u);
#endif
    const char* slash = strrchr(library_path, '/');
    const char* backslash = strrchr(library_path, '\\');
    if (backslash != NULL && (slash == NULL || backslash > slash)) slash = backslash;
    if (slash == NULL) return EUHEDRAL_CUDA_KERNEL_UNAVAILABLE;
    char path[PATH_MAX];
    int length = snprintf(path, sizeof(path), "%.*s/../share/euhedral_cuda/%s",
            (int)(slash - library_path), library_path, source_name);
    if (length < 0 || (size_t)length >= sizeof(path)) return EUHEDRAL_CUDA_KERNEL_UNAVAILABLE;
    FILE* file = fopen(path, "rb");
    if (file == NULL) {
        fprintf(stderr, "Euhedral CUDA: cannot open %s: %s\n", path, strerror(errno));
        return EUHEDRAL_CUDA_KERNEL_UNAVAILABLE;
    }
    if (fseek(file, 0, SEEK_END) != 0) { fclose(file); return EUHEDRAL_CUDA_KERNEL_UNAVAILABLE; }
    long size = ftell(file);
    if (size <= 0) { fclose(file); return EUHEDRAL_CUDA_KERNEL_UNAVAILABLE; }
    rewind(file);
    char* source = malloc((size_t)size + 1u);
    if (source == NULL) { fclose(file); return EUHEDRAL_CUDA_KERNEL_UNAVAILABLE; }
    size_t read_size = fread(source, 1, (size_t)size, file);
    fclose(file);
    if (read_size != (size_t)size) { free(source); return EUHEDRAL_CUDA_KERNEL_UNAVAILABLE; }
    source[size] = '\0';

    nvrtcProgram program = NULL;
    nvrtcResult nv_status = nvrtcCreateProgram(&program, source, source_name, 0, NULL, NULL);
    free(source);
    if (nv_status != NVRTC_SUCCESS) return EUHEDRAL_CUDA_KERNEL_UNAVAILABLE;
    const char* options[] = {"--std=c++14", "--gpu-architecture=compute_90"};
    nv_status = nvrtcCompileProgram(program, 2, options);
    if (nv_status != NVRTC_SUCCESS) {
        size_t log_size = 0;
        nvrtcGetProgramLogSize(program, &log_size);
        if (log_size > 0) {
            char* log = malloc(log_size);
            if (log != NULL) { nvrtcGetProgramLog(program, log); fprintf(stderr, "%s\n", log); free(log); }
        }
        nvrtcDestroyProgram(&program);
        return EUHEDRAL_CUDA_KERNEL_UNAVAILABLE;
    }
    size_t ptx_size = 0;
    nv_status = nvrtcGetPTXSize(program, &ptx_size);
    char* ptx = nv_status == NVRTC_SUCCESS ? malloc(ptx_size) : NULL;
    if (ptx == NULL || nv_status != NVRTC_SUCCESS || nvrtcGetPTX(program, ptx) != NVRTC_SUCCESS) {
        free(ptx); nvrtcDestroyProgram(&program); return EUHEDRAL_CUDA_KERNEL_UNAVAILABLE;
    }
    nvrtcDestroyProgram(&program);
    CUresult status = cuInit(0);
    CUcontext context = NULL;
    if (status == CUDA_SUCCESS) status = cuCtxGetCurrent(&context);
    if (status == CUDA_SUCCESS && context == NULL) status = CUDA_ERROR_INVALID_CONTEXT;
    if (status == CUDA_SUCCESS) status = cuModuleLoadData(module, ptx);
    free(ptx);
    if (status == CUDA_SUCCESS) status = cuModuleGetFunction(function, *module, function_name);
    if (status != CUDA_SUCCESS && *module != NULL) {
        (void)cuModuleUnload(*module);
        *module = NULL;
    }
    return status == CUDA_SUCCESS ? EUHEDRAL_CUDA_SUCCESS : EUHEDRAL_CUDA_KERNEL_UNAVAILABLE;
}
