import org.gradle.api.plugins.JavaPluginExtension
import org.gradle.api.tasks.testing.Test
import org.gradle.jvm.toolchain.JavaLanguageVersion

// Common Java configuration applied to every subproject that applies the `java` plugin.
subprojects {
    plugins.withId("java") {
        the<JavaPluginExtension>().toolchain {
            languageVersion = JavaLanguageVersion.of(25)
        }
        tasks.withType<Test>().configureEach {
            useJUnitPlatform()
        }
    }
}