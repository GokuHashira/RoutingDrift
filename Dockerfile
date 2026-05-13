# RoutingDrift — MSML 605
# Requires: docker run --gpus all
# Model weights are NOT baked in — mount them at runtime and set OLMOE_PATH / MIXTRAL_PATH via .env

FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/cache/huggingface \
    TRITON_CACHE_DIR=/tmp/triton_cache \
    PYTHONPATH=/workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 \
        python3.10-dev \
        python3-pip \
        build-essential \
        git \
        curl && \
    rm -rf /var/lib/apt/lists/* && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/pip    pip    /usr/bin/pip3    1

RUN pip install --upgrade pip setuptools wheel

WORKDIR /workspace

RUN pip install "torch>=2.3.0" --index-url https://download.pytorch.org/whl/cu121

COPY requirements.txt .
RUN pip install -r requirements.txt

RUN pip install \
    "python-dotenv>=1.0.0" \
    "optimum>=1.19.0" \
    "auto-gptq>=0.7.0"

# ── Source code ───────────────────────────────────────────────────────────────
COPY . .

RUN rm -f .env

RUN chmod +x docker-entrypoint.sh

VOLUME ["/cache/huggingface", "/workspace/results"]

ENTRYPOINT ["/workspace/docker-entrypoint.sh"]

CMD []
