# syntax=docker/dockerfile:1
FROM eclipse-temurin:25-jdk-noble AS build
WORKDIR /src
RUN apt-get update && apt-get install -y --no-install-recommends curl xz-utils ca-certificates \
    && rm -rf /var/lib/apt/lists/*
ARG ZIG_SHA256=70e49664a74374b48b51e6f3fdfbf437f6395d42509050588bd49abe52ba3d00
RUN curl -fsSL https://ziglang.org/download/0.16.0/zig-x86_64-linux-0.16.0.tar.xz -o /tmp/zig.tar.xz \
    && printf '%s  %s\n' "$ZIG_SHA256" /tmp/zig.tar.xz | sha256sum -c - \
    && mkdir -p /opt/zig \
    && tar -xJf /tmp/zig.tar.xz --strip-components=1 -C /opt/zig \
    && rm /tmp/zig.tar.xz
ENV ZIG=/opt/zig/zig
COPY . .
RUN ./gradlew :api:bootJar -Peuhedral.native.products=linux-x64 \
    -Peuhedral.cuda.force-download=true --no-daemon
RUN "$JAVA_HOME/bin/jlink" --add-modules \
    java.base,java.compiler,java.desktop,java.instrument,java.logging,java.management,java.naming,java.net.http,java.prefs,java.security.jgss,java.security.sasl,java.sql,java.xml,jdk.crypto.ec,jdk.jfr,jdk.management,jdk.unsupported \
    --strip-debug --no-man-pages --no-header-files --compress=zip-6 --output /opt/euhedral-jre \
    && "$JAVA_HOME/bin/javac" -d /opt/health container/HealthProbe.java

FROM gcr.io/distroless/cc-debian13:nonroot
WORKDIR /opt/euhedral
COPY --from=build /opt/euhedral-jre/ /opt/euhedral/jre/
COPY --from=build /opt/health/ /opt/euhedral/health/
COPY --from=build /src/api/build/libs/euhedral-inference-api.jar /opt/euhedral/app.jar
COPY --from=build /src/build/native/linux-x64/lib/libeuhedral_cuda.so /opt/euhedral/lib/libeuhedral_cuda.so
COPY --from=build /src/build/native/linux-x64/share/euhedral_cuda/ /opt/euhedral/share/euhedral_cuda/
COPY --from=build /src/build/cuda-dev/linux-x64/runtime/ /opt/euhedral/lib/
COPY --from=build /src/build/cuda-dev/linux-x64/include/ /opt/euhedral/cuda/include/
ENV LD_LIBRARY_PATH=/opt/euhedral/lib \
    EUHEDRAL_CUDA_INCLUDE_DIR=/opt/euhedral/cuda/include \
    EUHEDRAL_INFERENCE_CUDA_LIBRARY_PATH=/opt/euhedral/lib/libeuhedral_cuda.so \
    EUHEDRAL_INFERENCE_ARTIFACT_PATH=/models/model.edrl \
    EUHEDRAL_INFERENCE_TOKENIZER_DIRECTORY=/tokenizer \
    EUHEDRAL_INFERENCE_WORKER_CPUS=0 \
    EUHEDRAL_INFERENCE_MODEL_ID=qwen \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD ["/opt/euhedral/jre/bin/java", "-cp", "/opt/euhedral/health", "HealthProbe"]
ENTRYPOINT ["/opt/euhedral/jre/bin/java", "--enable-native-access=ALL-UNNAMED", "-jar", "/opt/euhedral/app.jar"]
