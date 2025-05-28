FROM rust:1.83 as builder

# Install ONNX Runtime in the builder stage
RUN wget -q https://github.com/microsoft/onnxruntime/releases/download/v1.16.0/onnxruntime-linux-x64-1.16.0.tgz \
    && tar -xzf onnxruntime-linux-x64-1.16.0.tgz \
    && cp -r onnxruntime-linux-x64-1.16.0/lib/* /usr/local/lib/ \
    && cp -r onnxruntime-linux-x64-1.16.0/include/* /usr/local/include/ \
    && ldconfig \
    && rm -rf onnxruntime-*

ENV ORT_LIB_LOCATION=/usr/local/lib

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

# Install the exact same ONNX Runtime version
RUN wget -q https://github.com/microsoft/onnxruntime/releases/download/v1.16.0/onnxruntime-linux-x64-1.16.0.tgz \
    && tar -xzf onnxruntime-linux-x64-1.16.0.tgz \
    && cp onnxruntime-linux-x64-1.16.0/lib/* /usr/lib/ \
    && ldconfig \
    && rm -rf onnxruntime-*

WORKDIR /app
COPY --from=builder /app/target/release/rust-api .

EXPOSE 8001

CMD ["./rust-api"]