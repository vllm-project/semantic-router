module github.com/vllm-project/semantic-router/dashboard/backend

go 1.24.1

require (
	github.com/google/uuid v1.6.0
	github.com/mattn/go-sqlite3 v1.14.33
	github.com/vllm-project/semantic-router/src/semantic-router v0.0.0
	gopkg.in/yaml.v3 v3.0.1
)

require (
	go.uber.org/multierr v1.11.0 // indirect
	go.uber.org/zap v1.27.0 // indirect
	gopkg.in/yaml.v2 v2.4.0 // indirect
)

replace github.com/vllm-project/semantic-router/src/semantic-router => ../../src/semantic-router
