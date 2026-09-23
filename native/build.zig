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
    library.root_module.addIncludePath(b.path("include"));
    library.root_module.addIncludePath(.{.cwd_relative = cuda_include_dir});
    library.root_module.addLibraryPath(.{.cwd_relative = cuda_lib_dir});
    library.root_module.linkSystemLibrary("cudart", .{});
    b.installArtifact(library);
}
