# A router image whose model runtime has this plugin, for
# `vllm-sr serve --image`. From the repository root:
#
#   docker build -t vllm-sr-example src/model-runtime/examples/third_party_plugin
#
# IMAGE selects the router image to extend (vllm-sr-rocm or vllm-sr-cuda for a
# GPU platform).
ARG IMAGE=ghcr.io/vllm-project/semantic-router/vllm-sr:latest
FROM ${IMAGE}
COPY . /tmp/plugin
RUN pip install --no-deps /tmp/plugin && rm -rf /tmp/plugin
