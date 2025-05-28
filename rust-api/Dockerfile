FROM rust:1.83 as builder

WORKDIR /app
COPY Cargo.toml .
COPY src/ src/
RUN cargo build --release

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y \
    libssl3 \
    ca-certificates \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install ONNX Runtime with correct symlinks
RUN wget -q https://github.com/microsoft/onnxruntime/releases/download/v1.16.3/onnxruntime-linux-x64-1.16.3.tgz \
    && tar -xzf onnxruntime-linux-x64-1.16.3.tgz \
    && cp onnxruntime-linux-x64-1.16.3/lib/* /usr/lib/ \
    && ln -sf /usr/lib/libonnxruntime.so.1.16.3 /usr/lib/libonnxruntime.so.1.16.0 \
    && ln -sf /usr/lib/libonnxruntime.so.1.16.3 /usr/lib/libonnxruntime.so.1 \
    && ln -sf /usr/lib/libonnxruntime.so.1.16.3 /usr/lib/libonnxruntime.so \
    && ldconfig \
    && rm -rf onnxruntime-*

WORKDIR /app
COPY --from=builder /app/target/release/rust-api .

EXPOSE 8001

CMD ["./rust-api"]