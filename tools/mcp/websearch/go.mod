module github.com/vllm-project/semantic-router/tools/mcp/websearch

go 1.25.0

replace github.com/vllm-project/semantic-router/src/semantic-router => ../../../src/semantic-router

require (
	github.com/mark3labs/mcp-go v0.43.2
	github.com/vllm-project/semantic-router/src/semantic-router v0.0.0-00010101000000-000000000000
)

require (
	github.com/bahlo/generic-list-go v0.2.0 // indirect
	github.com/buger/jsonparser v1.1.2 // indirect
	github.com/google/uuid v1.6.0 // indirect
	github.com/invopop/jsonschema v0.13.0 // indirect
	github.com/mailru/easyjson v0.7.7 // indirect
	github.com/spf13/cast v1.7.1 // indirect
	github.com/wk8/go-ordered-map/v2 v2.1.8 // indirect
	github.com/yosida95/uritemplate/v3 v3.0.2 // indirect
	gopkg.in/yaml.v3 v3.0.1 // indirect
)
