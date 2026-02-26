module semantic-router/ml-model-selection

go 1.24.1

replace github.com/vllm-project/semantic-router/candle-binding => ../../../candle-binding

replace github.com/vllm-project/semantic-router/ml-binding => ../../../ml-binding

replace github.com/vllm-project/semantic-router/src/semantic-router => ../../semantic-router

require (
	github.com/vllm-project/semantic-router/candle-binding v0.0.0-00010101000000-000000000000
	github.com/vllm-project/semantic-router/src/semantic-router v0.0.0-00010101000000-000000000000
)

require (
	github.com/vllm-project/semantic-router/ml-binding v0.0.0-00010101000000-000000000000 // indirect
	go.uber.org/multierr v1.11.0 // indirect
	go.uber.org/zap v1.27.0 // indirect
	gopkg.in/yaml.v2 v2.4.0 // indirect
	gopkg.in/yaml.v3 v3.0.1 // indirect
)
