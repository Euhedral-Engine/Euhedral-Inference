#include "cuda_kernel_loader.h"
#include "euhedral_cuda.h"
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
static int quantized_anchor;
static int gdn_anchor;
static int elementwise_anchor;
static CUmodule quantized_module;
static CUmodule gdn_module;
static CUmodule elementwise_module;
static CUfunction linear_quantized;
static CUfunction linear_bf16_to_float;
static CUfunction gdn_control;
static CUfunction gdn_convolution;
static CUfunction gdn_recurrence;
static CUfunction gdn_gated_rms_norm;
static CUfunction residual_add;
static CUfunction swiglu;
static int init_status = EUHEDRAL_CUDA_KERNEL_UNAVAILABLE;

static int get_function(CUmodule module, CUfunction* function, const char* name) {
    return (int)cuModuleGetFunction(function, module, name);
}

static void initialize(void) {
    init_status = euhedral_cuda_load_kernel(
            &quantized_anchor, "qwen_layer_linear.cu", "euhedral_linear_quantized_bf16", &quantized_module, &linear_quantized);
    if (init_status != EUHEDRAL_CUDA_SUCCESS) return;
    CUresult status = get_function(quantized_module, &linear_bf16_to_float, "euhedral_linear_bf16_to_float");
    if (status != CUDA_SUCCESS) { init_status = (int)status; return; }

    init_status = euhedral_cuda_load_kernel(
            &gdn_anchor, "qwen_gdn_ops.cu", "euhedral_gdn_control_fp32", &gdn_module, &gdn_control);
    if (init_status != EUHEDRAL_CUDA_SUCCESS) return;
    status = get_function(gdn_module, &gdn_convolution, "euhedral_gdn_convolution_bf16");
    if (status != CUDA_SUCCESS) { init_status = (int)status; return; }
    status = get_function(gdn_module, &gdn_recurrence, "euhedral_gdn_recurrence_bf16");
    if (status != CUDA_SUCCESS) { init_status = (int)status; return; }
    status = get_function(gdn_module, &gdn_gated_rms_norm, "euhedral_gdn_gated_rms_norm_bf16");
    if (status != CUDA_SUCCESS) { init_status = (int)status; return; }

    init_status = euhedral_cuda_load_kernel(
            &elementwise_anchor, "qwen_elementwise.cu", "euhedral_residual_add_bf16", &elementwise_module, &residual_add);
    if (init_status != EUHEDRAL_CUDA_SUCCESS) return;
    status = get_function(elementwise_module, &swiglu, "euhedral_swiglu_bf16");
    init_status = status == CUDA_SUCCESS ? EUHEDRAL_CUDA_SUCCESS : (int)status;
}

#ifdef _WIN32
static BOOL CALLBACK initialize_once(PINIT_ONCE state, PVOID parameter, PVOID* context) {
    (void)state; (void)parameter; (void)context;
    initialize();
    return TRUE;
}
#endif

static int ensure_initialized(void) {
#ifdef _WIN32
    if (!InitOnceExecuteOnce(&once, initialize_once, NULL, NULL)) return EUHEDRAL_CUDA_KERNEL_UNAVAILABLE;
#else
    if (pthread_once(&once, initialize) != 0) return EUHEDRAL_CUDA_KERNEL_UNAVAILABLE;
#endif
    return init_status;
}

static int launch_and_synchronize(CUfunction function, uint32_t grid_x, uint32_t block_x, void** parameters) {
    CUresult status = cuLaunchKernel(function, grid_x, 1, 1, block_x, 1, 1, 0, NULL, parameters, NULL);
    if (status != CUDA_SUCCESS) return (int)status;
    cudaError_t sync = cudaDeviceSynchronize();
    return sync == cudaSuccess ? EUHEDRAL_CUDA_SUCCESS : (int)sync;
}

static int align_plane(uint64_t size, uint64_t* aligned) {
    if (size > UINT64_MAX - 255) return EUHEDRAL_CUDA_SIZE_OVERFLOW;
    *aligned = (size + 255) & ~UINT64_C(255);
    return EUHEDRAL_CUDA_SUCCESS;
}

static int quantized_byte_size(uint32_t in_features, uint32_t out_features, uint32_t bits, uint64_t* required) {
    if (in_features == 0 || in_features % 128 != 0 || out_features == 0 || (bits != 4 && bits != 5))
        return EUHEDRAL_CUDA_FORMAT_MISMATCH;
    const uint64_t groups = in_features / 64;
    if ((uint64_t)out_features > UINT64_MAX / groups / 32) return EUHEDRAL_CUDA_SIZE_OVERFLOW;
    const uint64_t code_bytes = (uint64_t)out_features * groups * 32;
    uint64_t code_plane_bytes;
    int status = align_plane(code_bytes, &code_plane_bytes);
    if (status != EUHEDRAL_CUDA_SUCCESS) return status;
    uint64_t high_plane_bytes = 0;
    if (bits == 5) {
        if ((uint64_t)out_features > UINT64_MAX / groups / 8) return EUHEDRAL_CUDA_SIZE_OVERFLOW;
        const uint64_t high_bytes = (uint64_t)out_features * groups * 8;
        status = align_plane(high_bytes, &high_plane_bytes);
        if (status != EUHEDRAL_CUDA_SUCCESS) return status;
    }
    if (code_plane_bytes > UINT64_MAX - high_plane_bytes) return EUHEDRAL_CUDA_SIZE_OVERFLOW;
    const uint64_t scale_offset = code_plane_bytes + high_plane_bytes;
    if ((uint64_t)out_features > (UINT64_MAX - scale_offset) / groups / 2) return EUHEDRAL_CUDA_SIZE_OVERFLOW;
    *required = scale_offset + (uint64_t)out_features * groups * 2;
    return EUHEDRAL_CUDA_SUCCESS;
}

int euhedral_cuda_linear_quantized_bf16(
        const void* device_input, const void* device_weights, void* device_output,
        uint32_t rows, uint32_t in_features, uint32_t out_features,
        uint64_t weights_byte_size, uint32_t bits) {
    if (device_input == NULL || device_weights == NULL || device_output == NULL || rows == 0 || out_features == 0)
        return EUHEDRAL_CUDA_INVALID_ARGUMENT;
    uint64_t required;
    int status = quantized_byte_size(in_features, out_features, bits, &required);
    if (status != EUHEDRAL_CUDA_SUCCESS) return status;
    if (required != weights_byte_size) return EUHEDRAL_CUDA_FORMAT_MISMATCH;
    const uint64_t count = (uint64_t)rows * out_features;
    if (count > UINT32_MAX) return EUHEDRAL_CUDA_SIZE_OVERFLOW;
    status = euhedral_cuda_bind_thread_context();
    if (status != EUHEDRAL_CUDA_SUCCESS) return status;
    status = ensure_initialized();
    if (status != EUHEDRAL_CUDA_SUCCESS) return status;
    CUdeviceptr input = (CUdeviceptr)(uintptr_t)device_input;
    CUdeviceptr weights = (CUdeviceptr)(uintptr_t)device_weights;
    CUdeviceptr output = (CUdeviceptr)(uintptr_t)device_output;
    uint32_t rows_arg = rows, in_arg = in_features, out_arg = out_features, bits_arg = bits;
    void* parameters[] = {&input, &weights, &output, &rows_arg, &in_arg, &out_arg, &bits_arg};
    return launch_and_synchronize(linear_quantized, (uint32_t)count, 128, parameters);
}

int euhedral_cuda_linear_bf16_to_float(
        const void* device_input, const void* device_weights, void* device_output,
        uint32_t rows, uint32_t in_features, uint32_t out_features) {
    if (device_input == NULL || device_weights == NULL || device_output == NULL || rows == 0 || in_features == 0 || out_features == 0)
        return EUHEDRAL_CUDA_INVALID_ARGUMENT;
    const uint64_t count = (uint64_t)rows * out_features;
    if (count > UINT32_MAX) return EUHEDRAL_CUDA_SIZE_OVERFLOW;
    int status = euhedral_cuda_bind_thread_context();
    if (status != EUHEDRAL_CUDA_SUCCESS) return status;
    status = ensure_initialized();
    if (status != EUHEDRAL_CUDA_SUCCESS) return status;
    CUdeviceptr input = (CUdeviceptr)(uintptr_t)device_input;
    CUdeviceptr weights = (CUdeviceptr)(uintptr_t)device_weights;
    CUdeviceptr output = (CUdeviceptr)(uintptr_t)device_output;
    uint32_t rows_arg = rows, in_arg = in_features, out_arg = out_features;
    void* parameters[] = {&input, &weights, &output, &rows_arg, &in_arg, &out_arg};
    return launch_and_synchronize(linear_bf16_to_float, (uint32_t)count, 128, parameters);
}

int euhedral_cuda_gdn_control_fp32(
        const float* device_a_projection, const float* device_b_projection,
        const float* device_a_log, const float* device_dt_bias,
        float* device_g_output, float* device_beta_output, uint32_t rows, uint32_t heads) {
    if (device_a_projection == NULL || device_b_projection == NULL || device_a_log == NULL || device_dt_bias == NULL
            || device_g_output == NULL || device_beta_output == NULL || rows == 0 || heads == 0)
        return EUHEDRAL_CUDA_INVALID_ARGUMENT;
    const uint64_t count = (uint64_t)rows * heads;
    if (count > UINT32_MAX) return EUHEDRAL_CUDA_SIZE_OVERFLOW;
    int status = euhedral_cuda_bind_thread_context();
    if (status != EUHEDRAL_CUDA_SUCCESS) return status;
    status = ensure_initialized();
    if (status != EUHEDRAL_CUDA_SUCCESS) return status;
    CUdeviceptr a = (CUdeviceptr)(uintptr_t)device_a_projection;
    CUdeviceptr b = (CUdeviceptr)(uintptr_t)device_b_projection;
    CUdeviceptr a_log = (CUdeviceptr)(uintptr_t)device_a_log;
    CUdeviceptr dt_bias = (CUdeviceptr)(uintptr_t)device_dt_bias;
    CUdeviceptr g = (CUdeviceptr)(uintptr_t)device_g_output;
    CUdeviceptr beta = (CUdeviceptr)(uintptr_t)device_beta_output;
    uint32_t rows_arg = rows, heads_arg = heads;
    void* parameters[] = {&a, &b, &a_log, &dt_bias, &g, &beta, &rows_arg, &heads_arg};
    return launch_and_synchronize(gdn_control, (uint32_t)((count + 127) / 128), 128, parameters);
}

int euhedral_cuda_gdn_convolution_bf16(
        const void* device_query_key, const void* device_value_z, const void* device_convolution_weights,
        void* device_convolution_state, void* device_output, uint32_t rows,
        uint32_t query_key_width, uint32_t value_width, uint32_t convolution_width, uint32_t kernel_size) {
    if (device_query_key == NULL || device_value_z == NULL || device_convolution_weights == NULL
            || device_convolution_state == NULL || device_output == NULL || rows == 0 || query_key_width == 0
            || value_width == 0 || convolution_width != query_key_width + value_width || kernel_size < 2 || kernel_size > 32)
        return EUHEDRAL_CUDA_INVALID_ARGUMENT;
    int status = euhedral_cuda_bind_thread_context();
    if (status != EUHEDRAL_CUDA_SUCCESS) return status;
    status = ensure_initialized();
    if (status != EUHEDRAL_CUDA_SUCCESS) return status;
    CUdeviceptr qk = (CUdeviceptr)(uintptr_t)device_query_key;
    CUdeviceptr vz = (CUdeviceptr)(uintptr_t)device_value_z;
    CUdeviceptr weights = (CUdeviceptr)(uintptr_t)device_convolution_weights;
    CUdeviceptr state = (CUdeviceptr)(uintptr_t)device_convolution_state;
    CUdeviceptr output = (CUdeviceptr)(uintptr_t)device_output;
    uint32_t rows_arg = rows, qk_arg = query_key_width, value_arg = value_width;
    uint32_t channels_arg = convolution_width, kernel_arg = kernel_size;
    void* parameters[] = {&qk, &vz, &weights, &state, &output, &rows_arg, &qk_arg, &value_arg, &channels_arg, &kernel_arg};
    return launch_and_synchronize(gdn_convolution, (convolution_width + 127) / 128, 128, parameters);
}

int euhedral_cuda_gdn_recurrence_bf16(
        const void* device_convolved, const float* device_g, const float* device_beta,
        float* device_recurrent_state, void* device_output, uint32_t rows,
        uint32_t key_heads, uint32_t value_heads, uint32_t key_head_dim,
        uint32_t value_head_dim, float output_scale) {
    if (device_convolved == NULL || device_g == NULL || device_beta == NULL || device_recurrent_state == NULL
            || device_output == NULL || rows == 0 || key_heads == 0 || value_heads == 0
            || value_heads % key_heads != 0 || key_head_dim != 128 || value_head_dim != 128
            || !isfinite(output_scale) || output_scale <= 0.0f)
        return EUHEDRAL_CUDA_INVALID_ARGUMENT;
    const uint64_t rows_count = (uint64_t)value_heads * value_head_dim;
    if (rows_count > UINT32_MAX) return EUHEDRAL_CUDA_SIZE_OVERFLOW;
    int status = euhedral_cuda_bind_thread_context();
    if (status != EUHEDRAL_CUDA_SUCCESS) return status;
    status = ensure_initialized();
    if (status != EUHEDRAL_CUDA_SUCCESS) return status;
    CUdeviceptr convolved = (CUdeviceptr)(uintptr_t)device_convolved;
    CUdeviceptr g = (CUdeviceptr)(uintptr_t)device_g;
    CUdeviceptr beta = (CUdeviceptr)(uintptr_t)device_beta;
    CUdeviceptr state = (CUdeviceptr)(uintptr_t)device_recurrent_state;
    CUdeviceptr output = (CUdeviceptr)(uintptr_t)device_output;
    uint32_t rows_arg = rows, key_heads_arg = key_heads, value_heads_arg = value_heads;
    uint32_t key_dim_arg = key_head_dim, value_dim_arg = value_head_dim;
    void* parameters[] = {&convolved, &g, &beta, &state, &output, &rows_arg,
            &key_heads_arg, &value_heads_arg, &key_dim_arg, &value_dim_arg, &output_scale};
    return launch_and_synchronize(gdn_recurrence, (uint32_t)rows_count, key_head_dim, parameters);
}

int euhedral_cuda_gdn_gated_rms_norm_bf16(
        const void* device_recurrent, const void* device_value_z, const void* device_norm_weight,
        void* device_output, uint32_t rows, uint32_t value_heads, uint32_t head_dim, float epsilon) {
    if (device_recurrent == NULL || device_value_z == NULL || device_norm_weight == NULL || device_output == NULL
            || rows == 0 || value_heads == 0 || head_dim != 128 || !isfinite(epsilon) || epsilon < 0.0f)
        return EUHEDRAL_CUDA_INVALID_ARGUMENT;
    const uint64_t count = (uint64_t)rows * value_heads;
    if (count > UINT32_MAX) return EUHEDRAL_CUDA_SIZE_OVERFLOW;
    int status = euhedral_cuda_bind_thread_context();
    if (status != EUHEDRAL_CUDA_SUCCESS) return status;
    status = ensure_initialized();
    if (status != EUHEDRAL_CUDA_SUCCESS) return status;
    CUdeviceptr recurrent = (CUdeviceptr)(uintptr_t)device_recurrent;
    CUdeviceptr value_z = (CUdeviceptr)(uintptr_t)device_value_z;
    CUdeviceptr weight = (CUdeviceptr)(uintptr_t)device_norm_weight;
    CUdeviceptr output = (CUdeviceptr)(uintptr_t)device_output;
    uint32_t rows_arg = rows, heads_arg = value_heads, dim_arg = head_dim;
    void* parameters[] = {&recurrent, &value_z, &weight, &output, &rows_arg, &heads_arg, &dim_arg, &epsilon};
    return launch_and_synchronize(gdn_gated_rms_norm, (uint32_t)count, 128, parameters);
}

int euhedral_cuda_residual_add_bf16(
        const void* device_residual, const void* device_delta, void* device_output, uint32_t rows, uint32_t width) {
    if (device_residual == NULL || device_delta == NULL || device_output == NULL || rows == 0 || width == 0)
        return EUHEDRAL_CUDA_INVALID_ARGUMENT;
    const uint64_t count = (uint64_t)rows * width;
    if (count > UINT32_MAX) return EUHEDRAL_CUDA_SIZE_OVERFLOW;
    int status = euhedral_cuda_bind_thread_context();
    if (status != EUHEDRAL_CUDA_SUCCESS) return status;
    status = ensure_initialized();
    if (status != EUHEDRAL_CUDA_SUCCESS) return status;
    CUdeviceptr residual = (CUdeviceptr)(uintptr_t)device_residual;
    CUdeviceptr delta = (CUdeviceptr)(uintptr_t)device_delta;
    CUdeviceptr output = (CUdeviceptr)(uintptr_t)device_output;
    uint32_t count_arg = (uint32_t)count;
    void* parameters[] = {&residual, &delta, &output, &count_arg};
    return launch_and_synchronize(residual_add, (uint32_t)((count + 255) / 256), 256, parameters);
}

int euhedral_cuda_swiglu_bf16(
        const void* device_gate_up, void* device_output, uint32_t rows, uint32_t intermediate_size) {
    if (device_gate_up == NULL || device_output == NULL || rows == 0 || intermediate_size == 0)
        return EUHEDRAL_CUDA_INVALID_ARGUMENT;
    const uint64_t count = (uint64_t)rows * intermediate_size;
    if (count > UINT32_MAX) return EUHEDRAL_CUDA_SIZE_OVERFLOW;
    int status = euhedral_cuda_bind_thread_context();
    if (status != EUHEDRAL_CUDA_SUCCESS) return status;
    status = ensure_initialized();
    if (status != EUHEDRAL_CUDA_SUCCESS) return status;
    CUdeviceptr gate_up = (CUdeviceptr)(uintptr_t)device_gate_up;
    CUdeviceptr output = (CUdeviceptr)(uintptr_t)device_output;
    uint32_t rows_arg = rows, width_arg = intermediate_size;
    void* parameters[] = {&gate_up, &output, &rows_arg, &width_arg};
    return launch_and_synchronize(swiglu, (uint32_t)((count + 255) / 256), 256, parameters);
}

int euhedral_cuda_zero_device_memory(void* device_address, uint64_t byte_size) {
    if (device_address == NULL || byte_size == 0) return EUHEDRAL_CUDA_INVALID_ARGUMENT;
    int status = euhedral_cuda_bind_thread_context();
    if (status != EUHEDRAL_CUDA_SUCCESS) return status;
    cudaError_t result = cudaMemset(device_address, 0, (size_t)byte_size);
    if (result != cudaSuccess) return (int)result;
    result = cudaDeviceSynchronize();
    return result == cudaSuccess ? EUHEDRAL_CUDA_SUCCESS : (int)result;
}
