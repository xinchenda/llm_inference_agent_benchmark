FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive \
    LLM_BENCH_ROOT=/app \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git libcurl4-openssl-dev curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

ARG LLAMA_CPP_COMMIT=a83d68b562b19fba1f8836edc93eb76d9ca1ce25
RUN git clone https://github.com/ggml-org/llama.cpp /tmp/llama.cpp \
    && cd /tmp/llama.cpp \
    && git checkout "${LLAMA_CPP_COMMIT}" \
    && cmake -B build -DGGML_NATIVE=OFF -DGGML_METAL=OFF \
       -DBUILD_SHARED_LIBS=OFF -DLLAMA_CURL=ON -DLLAMA_BUILD_SERVER=ON \
    && cmake --build build --config Release --target llama-server -j2 \
    && cp build/bin/llama-server /usr/local/bin/llama-server \
    && rm -rf /tmp/llama.cpp

WORKDIR /app
COPY requirements.lock /app/requirements.lock
RUN python -m pip install --no-cache-dir \
      --timeout 300 --retries 10 \
      --index-url https://download.pytorch.org/whl/cpu \
      torch==2.15.0
RUN grep -v '^torch==' /app/requirements.lock > /tmp/requirements-cpu.lock \
    && python -m pip install --no-cache-dir --timeout 300 --retries 10 \
       -r /tmp/requirements-cpu.lock
COPY . /app
RUN python -m pip install --no-cache-dir --no-deps .

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"

CMD ["llm-bench", "serve", "--host", "0.0.0.0", "--port", "8000"]

