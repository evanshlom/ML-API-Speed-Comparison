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

# Install ONNX Runtime - simpler approach
RUN wget -q https://github.com/microsoft/onnxruntime/releases/download/v1.16.3/onnxruntime-linux-x64-1.16.3.tgz \
    && tar -xzf onnxruntime-linux-x64-1.16.3.tgz \
    && ls -la onnxruntime-linux-x64-1.16.3/ \
    && find onnxruntime-linux-x64-1.16.3 -name "*.so*" \
    && cp -r onnxruntime-linux-x64-1.16.3/lib/* /usr/local/lib/ 2>/dev/null || true \
    && ldconfig \
    && rm -rf onnxruntime-*

WORKDIR /app
COPY --from=builder /app/target/release/rust-api .

EXPOSE 8001

CMD ["./rust-api"]