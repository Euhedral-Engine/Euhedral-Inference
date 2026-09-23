const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const cuda_include_dir = b.option(
        []const u8,
        "cuda-include-dir",
        "CUDA include directory",
    ) orelse "/usr/local/cuda-13.1/targets/x86_64-linux/include";
    const cuda_lib_dir = b.option(
        []const u8,
        "cuda-lib-dir",
        "CUDA library directory",
    ) orelse "/usr/local/cuda-13.1/targets/x86_64-linux/lib";

    const library = b.addLibrary(.{
        .name = "euhedral_cuda",
        .root_module = b.createModule(.{
            .target = target,
            .optimize = optimize,
            .link_libc = true,
        }),
        .linkage = .dynamic,
    });
    library.root_module.addCSourceFile(.{
        .file = b.path("src/euhedral_cuda.c"),
        .flags = &.{"-std=c11", "-fvisibility=hidden"},
    });
    library.root_module.addCSourceFile(.{
        .file = b.path("src/q3_embedding.c"),
        .flags = &.{"-std=c11", "-fvisibility=hidden"},
    });
    library.root_module.addCSourceFile(.{
        .file = b.path("src/cuda_kernel_loader.c"),
        .flags = &.{"-std=c11", "-fvisibility=hidden"},
    });
    library.root_module.addCSourceFile(.{
        .file = b.path("src/rms_norm_bf16.c"),
        .flags = &.{"-std=c11", "-fvisibility=hidden"},
    });
    library.root_module.addCSourceFile(.{
        .file = b.path("src/q3_linear_bf16.c"),
        .flags = &.{"-std=c11", "-fvisibility=hidden"},
    });
    library.root_module.addCSourceFile(.{
        .file = b.path("src/qwen_layer_ops.c"),
        .flags = &.{"-std=c11", "-fvisibility=hidden"},
    });
    library.root_module.addIncludePath(b.path("include"));
    library.root_module.addIncludePath(.{.cwd_relative = cuda_include_dir});
    library.root_module.addLibraryPath(.{.cwd_relative = cuda_lib_dir});
    library.root_module.linkSystemLibrary("cudart", .{});
    library.root_module.linkSystemLibrary("nvrtc", .{});
    if (target.result.os.tag == .windows) {
        library.root_module.linkSystemLibrary("nvcuda", .{});
    } else {
        library.root_module.linkSystemLibrary("cuda", .{});
        library.root_module.linkSystemLibrary("pthread", .{});
        library.root_module.linkSystemLibrary("dl", .{});
    }
    b.installArtifact(library);
    b.installFile("src/q3_embedding.cu", "share/euhedral_cuda/q3_embedding.cu");
    b.installFile("src/rms_norm_bf16.cu", "share/euhedral_cuda/rms_norm_bf16.cu");
    b.installFile("src/q3_linear_bf16.cu", "share/euhedral_cuda/q3_linear_bf16.cu");
    b.installFile("src/qwen_layer_linear.cu", "share/euhedral_cuda/qwen_layer_linear.cu");
    b.installFile("src/qwen_gdn_ops.cu", "share/euhedral_cuda/qwen_gdn_ops.cu");
    b.installFile("src/qwen_elementwise.cu", "share/euhedral_cuda/qwen_elementwise.cu");
}
