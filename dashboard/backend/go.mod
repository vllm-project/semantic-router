module github.com/vllm-project/semantic-router/dashboard/backend

go 1.25.0

require (
	github.com/golang-jwt/jwt/v5 v5.2.2
	github.com/google/uuid v1.6.0
	github.com/gorilla/websocket v1.5.3
	github.com/mattn/go-sqlite3 v1.14.33
	github.com/vllm-project/semantic-router/src/semantic-router v0.0.0
	golang.org/x/crypto v0.52.0
	golang.org/x/sys v0.45.0
	gopkg.in/yaml.v3 v3.0.1
)

replace github.com/vllm-project/semantic-router/src/semantic-router => ../../src/semantic-router
