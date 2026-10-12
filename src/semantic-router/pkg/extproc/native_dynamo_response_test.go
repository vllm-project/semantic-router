package extproc

import (
	"context"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/gateway"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routing"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/upstream"
)

type nativeDynamoRand struct{}

func (nativeDynamoRand) Uint64() uint64 { return 0 }

// Inject a deterministic fallback choice while exercising the real upstream
// executor, gateway, routing engine and response session.
type nativeDynamoUpstream struct {
	set      *upstream.Set
	fallback bool
	result   *upstream.Result
}

func (u *nativeDynamoUpstream) Execute(ctx context.Context, call *routing.Call, listener string) (*upstream.Result, error) {
	if u.fallback {
		next := *call
		next.Route, next.Fallback = "secondary", nil
		call.Fallback = &nativeDynamoFallback{call: &next}
	}
	result, err := u.set.Execute(ctx, call, listener)
	u.result = result
	return result, err
}

type nativeDynamoFallback struct{ call *routing.Call }

func (f *nativeDynamoFallback) Next(context.Context, routing.Outcome) (routing.FallbackStep, error) {
	call := f.call
	f.call = nil
	return routing.FallbackStep{Call: call}, nil
}

func TestNativeGatewayDynamoResponseIdentity(t *testing.T) {
	for _, streamed := range []bool{false, true} {
		for _, mode := range []string{"direct", "retry", "fallback"} {
			for _, backendType := range []string{"dynamo", "vllm"} {
				t.Run(fmt.Sprintf("stream=%t/%s/%s", streamed, mode, backendType), func(t *testing.T) {
					good := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
						_, _ = io.Copy(io.Discard, r.Body)
						if streamed {
							w.Header().Set("Content-Type", "text/event-stream")
							_, _ = io.WriteString(w, "data: {\"id\":\"r1\",\"object\":\"chat.completion.chunk\",\"model\":\"provider-model\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"hello\"},\"finish_reason\":\"stop\"}],\"nvext\":{\"token_ids\":[1]}}\n\n")
							w.(http.Flusher).Flush()
							_, _ = io.WriteString(w, "data: [DONE]\n\n")
						} else {
							w.Header().Set("Content-Type", "application/json")
							_, _ = io.WriteString(w, `{"id":"r1","object":"chat.completion","model":"provider-model","choices":[{"index":0,"message":{"role":"assistant","content":"hello"},"finish_reason":"stop"}],"nvext":{"token_ids":[1]}}`)
						}
					}))
					defer good.Close()
					bad := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { w.WriteHeader(503) }))
					defer bad.Close()
					endpoint := func(name, url string) upstream.EndpointSpec {
						host, port, err := net.SplitHostPort(strings.TrimPrefix(url, "http://"))
						if err != nil {
							t.Fatal(err)
						}
						n, err := strconv.Atoi(port)
						if err != nil {
							t.Fatal(err)
						}
						return upstream.EndpointSpec{Name: name, Scheme: "http", Host: host, Port: n, Weight: 1}
					}
					actual, first := endpoint("actual", good.URL), endpoint("first", bad.URL)
					router, model := routingTestRouterForFormat(llmprotocol.OpenAIChatV1)
					// The originally planned backend has the opposite type. Only the serving
					// endpoint may authorize the response, not request routing or headers.
					opposite := "dynamo"
					if backendType == "dynamo" {
						opposite = "vllm"
					}
					router.Config.VLLMEndpoints[0].Type = opposite
					router.Config.VLLMEndpoints = append(router.Config.VLLMEndpoints,
						config.VLLMEndpoint{Name: actual.Name, Type: backendType},
						config.VLLMEndpoint{Name: first.Name, Type: opposite})
					primary := upstream.ClusterSpec{Name: model, Endpoints: []upstream.EndpointSpec{actual}, LBPolicy: upstream.LBRoundRobin}
					switch mode {
					case "retry":
						primary.Endpoints = []upstream.EndpointSpec{first, actual}
						primary.Policy.Retry = &upstream.RetryPolicy{NumRetries: 1, On: upstream.Retry5xx}
					case "fallback":
						primary.Endpoints = []upstream.EndpointSpec{first}
					}
					set, err := upstream.New(upstream.Topology{DefaultCluster: model, Clusters: []upstream.ClusterSpec{primary, {Name: "secondary", LBPolicy: upstream.LBRoundRobin, Endpoints: []upstream.EndpointSpec{actual}}}}, upstream.Options{Rand: nativeDynamoRand{}})
					if err != nil {
						t.Fatal(err)
					}
					defer func() { _ = set.Close(context.Background()) }()
					sender := &nativeDynamoUpstream{set: set, fallback: mode == "fallback"}
					opts := routing.DefaultOptions
					opts.ExecutesFallback = true
					handler, err := gateway.NewHandler(gateway.Options{Serving: gateway.Static(gateway.Serving{Engine: routing.NewEngine(router, opts), Upstream: sender})})
					if err != nil {
						t.Fatal(err)
					}
					request := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(fmt.Sprintf(`{"model":%q,"messages":[{"role":"user","content":"hello"}],"stream":%t}`, model, streamed)))
					request.Header.Set("Content-Type", "application/json")
					request.Header.Set("x-backend-type", "dynamo") // untrusted headers cannot authorize nvext
					response := httptest.NewRecorder()
					handler.ServeHTTP(response, request)
					if sender.result == nil || sender.result.Response.Endpoint.Name != "actual" {
						t.Fatalf("no actual upstream response: %s", response.Body.String())
					}
					if mode == "retry" && len(sender.result.Response.Attempts) != 2 {
						t.Fatalf("retry not exercised: %+v", sender.result.Response.Attempts)
					}
					if mode == "fallback" && sender.result.Hops != 2 {
						t.Fatalf("fallback not exercised: %+v", sender.result)
					}
					body := response.Body.String()
					if backendType == "dynamo" {
						if response.Code != 200 || !strings.Contains(body, `"nvext"`) || strings.Contains(body, "unexpected_dynamo") {
							t.Fatalf("valid response rejected: %d %s", response.Code, body)
						}
					} else {
						if strings.Contains(body, `"nvext"`) || strings.Contains(body, `"content":"hello"`) {
							t.Fatalf("invalid response leaked: %d %s", response.Code, body)
						}
						if streamed {
							if !strings.Contains(body, "unexpected_dynamo_nvext_backend") || strings.Contains(body, `"finish_reason":"stop"`) {
								t.Fatalf("stream did not fail safely: %s", body)
							}
						} else if response.Code != http.StatusBadGateway {
							t.Fatalf("buffered response status = %d, want 502: %s", response.Code, body)
						}
					}
				})
			}
		}
	}
}

func TestNativeBackendIdentityClearsPreviousPermission(t *testing.T) {
	router := dynamoBoundaryTestRouter("dynamo")
	ctx := &RequestContext{}
	router.captureNativeBackendIdentity("backend", ctx)
	if !ctx.AllowDynamoExtensions {
		t.Fatal("known Dynamo endpoint rejected")
	}
	for _, name := range []string{"", "unknown"} {
		ctx.AllowDynamoExtensions = true
		router.captureNativeBackendIdentity(name, ctx)
		if ctx.AllowDynamoExtensions || ctx.UpstreamBackendName != "" {
			t.Fatal("stale endpoint identity retained")
		}
	}
}
