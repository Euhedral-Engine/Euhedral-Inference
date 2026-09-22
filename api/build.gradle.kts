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

application {
    applicationName = "euhedral-inference-api"
    mainClass = "io.euhedral_execution.inference.api.Main"
}

dependencies {
    implementation(project(":core"))
}