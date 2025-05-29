FROM alpine:latest

RUN apk add --no-cache \
    wrk \
    curl \
    jq \
    bash \
    bc

WORKDIR /app

COPY wrk_scripts/ ./scripts/
COPY run_load_test.sh ./

RUN chmod +x run_load_test.sh

CMD ["./run_load_test.sh"]