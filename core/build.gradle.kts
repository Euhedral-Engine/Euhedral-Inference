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
    implementation(libs.jackson.databind)
    api(libs.slf4j.api)
    testImplementation(libs.junit.jupiter)
    testRuntimeOnly(libs.junit.platform.launcher)
}

tasks.named<Test>("test") {
    providers.gradleProperty("euhedral.cuda.async.library").orNull?.let {
        systemProperty("euhedral.cuda.async.library", it)
    }
}

val tokenizerReferenceDirectory = providers.gradleProperty("euhedral.qwen.tokenizer-dir")
    .orElse("/mnt/shared/qwen38-quant/source/qwen")

tasks.register<Test>("tokenizerReferenceTest") {
    group = "verification"
    description = "Verify Qwen tokenizer behavior against the selected checkpoint assets."
    val testSourceSet = sourceSets["test"]
    testClassesDirs = testSourceSet.output.classesDirs
    classpath = testSourceSet.runtimeClasspath
    include("**/QwenTokenizerTest.class")
    systemProperty("euhedral.qwen.tokenizer-dir", tokenizerReferenceDirectory.get())
    doFirst {
        val checkpoint = file(tokenizerReferenceDirectory.get())
        require(listOf("tokenizer.json", "tokenizer_config.json", "generation_config.json")
            .all { checkpoint.resolve(it).isFile }) {
            "Qwen tokenizer reference assets are required in $checkpoint; pass -Peuhedral.qwen.tokenizer-dir=..."
        }
    }
    useJUnitPlatform()
}