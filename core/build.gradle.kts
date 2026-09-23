plugins {
    `java-library`
    alias(libs.plugins.spotless)
}

spotless {
    java {
        palantirJavaFormat("2.96.0")
    }
}

dependencies {
    implementation(libs.euhedral.core)
    api(libs.slf4j.api)
    testImplementation(libs.junit.jupiter)
    testRuntimeOnly(libs.junit.platform.launcher)
}