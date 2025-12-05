module benchmark

go 1.24.1

toolchain go1.24.7

replace github.com/vllm-project/semantic-router/openvino-binding => ../..

replace github.com/vllm-project/semantic-router/candle-binding => ../../../candle-binding

require (
	github.com/vllm-project/semantic-router/candle-binding v0.0.0
	github.com/vllm-project/semantic-router/openvino-binding v0.0.0
)
