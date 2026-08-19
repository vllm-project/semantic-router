module semantic-router/ml-model-selection

go 1.25.0

replace github.com/vllm-project/semantic-router/candle-binding => ../../../../candle-binding

replace github.com/vllm-project/semantic-router/ml-binding => ../../../../ml-binding

replace github.com/vllm-project/semantic-router/src/semantic-router => ../../../semantic-router

require (
	github.com/vllm-project/semantic-router/candle-binding v0.0.0-00010101000000-000000000000
	github.com/vllm-project/semantic-router/src/semantic-router v0.0.0-00010101000000-000000000000
)

require (
	github.com/openai/openai-go v1.12.0 // indirect
	github.com/pemistahl/lingua-go v1.4.0 // indirect
	github.com/shopspring/decimal v1.3.1 // indirect
	github.com/tidwall/gjson v1.18.0 // indirect
	github.com/tidwall/match v1.1.1 // indirect
	github.com/tidwall/pretty v1.2.1 // indirect
	github.com/tidwall/sjson v1.2.5 // indirect
	github.com/vllm-project/semantic-router/ml-binding v0.0.0-00010101000000-000000000000 // indirect
	go.uber.org/multierr v1.11.0 // indirect
	go.uber.org/zap v1.27.0 // indirect
	golang.org/x/exp v0.0.0-20240719175910-8a7402abbf56 // indirect
	google.golang.org/protobuf v1.36.11 // indirect
	gopkg.in/yaml.v2 v2.4.0 // indirect
	gopkg.in/yaml.v3 v3.0.1 // indirect
)
