module github.com/vllm-project/semantic-router/dashboard/backend

go 1.24.1

require (
	github.com/chromedp/cdproto v0.0.0-20250724212937-08a3db8b4327
	github.com/chromedp/chromedp v0.14.2
	github.com/google/uuid v1.6.0
	github.com/gorilla/websocket v1.5.3
	github.com/mattn/go-sqlite3 v1.14.33
	github.com/vllm-project/semantic-router/src/semantic-router v0.0.0
	gopkg.in/yaml.v3 v3.0.1
)

require (
	github.com/chromedp/sysutil v1.1.0 // indirect
	github.com/go-json-experiment/json v0.0.0-20250725192818-e39067aee2d2 // indirect
	github.com/gobwas/httphead v0.1.0 // indirect
	github.com/gobwas/pool v0.2.1 // indirect
	github.com/gobwas/ws v1.4.0 // indirect
	go.uber.org/multierr v1.11.0 // indirect
	go.uber.org/zap v1.27.0 // indirect
	golang.org/x/sys v0.37.0 // indirect
	gopkg.in/yaml.v2 v2.4.0 // indirect
)

replace github.com/vllm-project/semantic-router/src/semantic-router => ../../src/semantic-router
