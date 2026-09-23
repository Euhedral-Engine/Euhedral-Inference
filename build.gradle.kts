import org.gradle.api.plugins.JavaPluginExtension
import org.gradle.api.tasks.Exec
import org.gradle.api.tasks.SourceSetContainer
import org.gradle.api.tasks.testing.Test
import org.gradle.jvm.toolchain.JavaLanguageVersion

val nativeBuildDirectory = layout.buildDirectory.dir("native")
val nativeLibraryFileName = when {
    System.getProperty("os.name").lowercase().contains("windows") -> "euhedral_cuda.dll"
    System.getProperty("os.name").lowercase().contains("mac") -> "libeuhedral_cuda.dylib"
    else -> "libeuhedral_cuda.so"
}

tasks.register<Exec>("nativeBuild") {
    group = "build"
    description = "Build the Euhedral CUDA native ABI with Zig."
    workingDir(rootProject.file("native"))
    doFirst {
        val includeDirectory = providers.gradleProperty("euhedral.cuda.include-dir").orNull
        val libraryDirectory = providers.gradleProperty("euhedral.cuda.library-dir").orNull
        require(includeDirectory != null && libraryDirectory != null) {
            "CUDA 13.1.x paths are required; pass " +
                    "-Peuhedral.cuda.include-dir=/path/to/cuda/include " +
                    "-Peuhedral.cuda.library-dir=/path/to/cuda/lib"
        }
        commandLine(
                "mise",
                "exec",
                "--",
                "zig",
                "build",
                "-Doptimize=Debug",
                "-Dcuda-include-dir=$includeDirectory",
                "-Dcuda-lib-dir=$libraryDirectory",
                "--prefix",
                nativeBuildDirectory.get().asFile.absolutePath)
    }
    inputs.dir(rootProject.file("native"))
    outputs.dir(nativeBuildDirectory)
}

// Common Java configuration applied to every subproject that applies the `java` plugin.
subprojects {
    plugins.withId("java") {
        the<JavaPluginExtension>().toolchain {
            languageVersion = JavaLanguageVersion.of(25)
        }
        tasks.named<Test>("test") {
            exclude("**/CudaGpuMemoryIntegrationTest.class")
            exclude("**/QwenCompactCudaResidencyIntegrationTest.class")
            exclude("**/QwenEmbeddingCudaIntegrationTest.class")
            exclude("**/CudaGpuOperationsIntegrationTest.class")
            exclude("**/QwenInstructionCudaIntegrationTest.class")
            exclude("**/QwenCompactCudaExecutionIntegrationTest.class")
            exclude("**/QwenLayerGpuOperationsIntegrationTest.class")
            exclude("**/QwenFirstLayerCudaIntegrationTest.class")
            useJUnitPlatform()
        }
        val testSourceSet = the<SourceSetContainer>()["test"]
        tasks.register<Test>("cudaIntegrationTest") {
            group = "verification"
            description = "Run dedicated CUDA 13.1.x memory, operator, residency, and execution integration tests."
            dependsOn(rootProject.tasks.named("nativeBuild"))
            testClassesDirs = testSourceSet.output.classesDirs
            classpath = testSourceSet.runtimeClasspath
            include("**/CudaGpuMemoryIntegrationTest.class")
            include("**/QwenCompactCudaResidencyIntegrationTest.class")
            include("**/QwenEmbeddingCudaIntegrationTest.class")
            include("**/CudaGpuOperationsIntegrationTest.class")
            include("**/QwenInstructionCudaIntegrationTest.class")
            include("**/QwenCompactCudaExecutionIntegrationTest.class")
            include("**/QwenLayerGpuOperationsIntegrationTest.class")
            include("**/QwenFirstLayerCudaIntegrationTest.class")
            systemProperty(
                    "euhedral.cuda.library",
                    nativeBuildDirectory.get().dir("lib").file(nativeLibraryFileName).asFile.absolutePath)
            systemProperty(
                    "euhedral.qwen.artifact",
                    providers.gradleProperty("euhedral.qwen.artifact")
                            .orElse("/mnt/shared/qwen38-quant/artifacts/qwen3_5_27b_compact_q3.edrl")
                            .get())
            systemProperty(
                    "euhedral.qwen.reference-artifact",
                    providers.gradleProperty("euhedral.qwen.reference-artifact")
                            .orElse("/mnt/shared/qwen38-quant/artifacts/qwen3_5_27b_bf16.edrl")
                            .get())
            jvmArgs("--enable-native-access=ALL-UNNAMED")
            doFirst {
                val includeDirectory = providers.gradleProperty("euhedral.cuda.include-dir").orNull
                val libraryDirectory = providers.gradleProperty("euhedral.cuda.library-dir").orNull
                require(includeDirectory != null && libraryDirectory != null) {
                    "CUDA 13.1.x header and runtime paths are required; pass " +
                            "-Peuhedral.cuda.include-dir=/path/to/cuda/include " +
                            "-Peuhedral.cuda.library-dir=/path/to/cuda/lib"
                }
                environment("EUHEDRAL_CUDA_INCLUDE_DIR", includeDirectory)
                val pathSeparator = System.getProperty("path.separator")
                val operatingSystem = System.getProperty("os.name").lowercase()
                if (operatingSystem.contains("windows")) {
                    environment(
                            "PATH",
                            libraryDirectory + pathSeparator + (System.getenv("PATH") ?: ""))
                } else {
                    environment(
                            "LD_LIBRARY_PATH",
                            libraryDirectory + pathSeparator + (System.getenv("LD_LIBRARY_PATH") ?: ""))
                }
            }
            useJUnitPlatform()
        }
    }
}