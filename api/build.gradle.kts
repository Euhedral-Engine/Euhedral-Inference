plugins {
    java
    application
}

application {
    applicationName = "euhedral-inference-api"
    mainClass = "io.euhedral_execution.inference.api.Main"
}

dependencies {
    implementation(project(":core"))
}