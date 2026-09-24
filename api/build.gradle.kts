plugins {
    java
    application
    alias(libs.plugins.spring.boot)
    alias(libs.plugins.spotless)
}

spotless {
    java {
        palantirJavaFormat("2.96.0")
    }
}

application {
    applicationName = "euhedral-inference-api"
    mainClass = "io.euhedral_execution.inference.api.EuhedralInferenceApplication"
    // The CUDA backend binds its native library through the FFM API.
    applicationDefaultJvmArgs = listOf("--enable-native-access=ALL-UNNAMED")
}

dependencies {
    implementation(platform(libs.spring.boot.dependencies))
    implementation(project(":core"))
    implementation(libs.spring.boot.starter.webmvc)
    testImplementation(platform(libs.spring.boot.dependencies))
    testImplementation(libs.spring.boot.starter.webmvc.test)
    testImplementation(libs.euhedral.core)
    testRuntimeOnly("org.junit.platform:junit-platform-launcher")
}

tasks.bootJar {
    manifest {
        attributes("Enable-Native-Access" to "ALL-UNNAMED")
    }
}

tasks.bootRun {
    jvmArgs("--enable-native-access=ALL-UNNAMED")
}
