import java.io.File

plugins {
    java
    application
    alias(libs.plugins.spotless)
}

spotless {
    java {
        palantirJavaFormat("2.96.0")
    }
}

// Benchmark harness only: not a dependency of production `core` or `api`, and not packaged in the API JAR.
application {
    applicationName = "euhedral-inference-benchmark"
    mainClass = "io.euhedral_execution.inference.benchmark.BenchmarkMain"
    applicationDefaultJvmArgs = listOf("--enable-native-access=ALL-UNNAMED")
}

dependencies {
    implementation(project(":core"))
    implementation(libs.jackson.databind)
    testImplementation(libs.junit.jupiter)
    testRuntimeOnly(libs.junit.platform.launcher)
}

val hostProductId = rootProject.extra["euhedral.native.host.id"] as String
val hostIncludeDirectory = rootProject.extra["euhedral.native.host.include"] as String
val hostRuntimeDirectory = rootProject.extra["euhedral.native.host.runtime"] as String

// Matches cudaIntegrationTest: build the host native library and expose its CUDA runtime/NVRTC inputs.
tasks.named<JavaExec>("run") {
    dependsOn(rootProject.tasks.named("nativeBuild${hostProductId.split("-").joinToString("") { it.replaceFirstChar(Char::uppercase) }}"))
    workingDir = rootProject.projectDir
    environment("EUHEDRAL_CUDA_INCLUDE_DIR", hostIncludeDirectory)
    val searchVariable = if (System.getProperty("os.name").startsWith("Windows")) "PATH" else "LD_LIBRARY_PATH"
    environment(searchVariable, hostRuntimeDirectory + File.pathSeparator + (System.getenv(searchVariable) ?: ""))
}
