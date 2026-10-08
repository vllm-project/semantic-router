package extensiontest

import (
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/extproc"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/gateway"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/pluginruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routing"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/upstream"
)

const overreachType = "test_overreach"

// refusedHeaders would re-route the request behind Envoy (the route key),
// relabel it as the Router, or break its framing.
var refusedHeaders = []string{
	"x-selected-model", "x-vsr-selected-decision", "x-envoy-original-path", "connection", "host", "content-length",
}

// overreachPlugin tries to write the headers the Router owns.
type overreachPlugin struct{}

func (*overreachPlugin) OnRequest(_ context.Context, request *pluginruntime.PluginRequest) error {
	refused := 0
	for _, name := range refusedHeaders {
		if request.SetHeader(name, "overreach") != nil {
			refused++
		}
	}
	return request.SetHeader("x-overreach-refused", strconv.Itoa(refused))
}

func (*overreachPlugin) OnResponse(_ context.Context, response *pluginruntime.PluginResponse) error {
	if response.SetHeader("x-vsr-response-path", "overreach") == nil {
		return errors.New("the plugin relabelled the response path")
	}
	return response.SetHeader("x-overreach-response", "refused")
}

func init() {
	if err := config.RegisterDecisionPlugin(config.NewDecisionPluginType(
		config.DecisionPluginCatalogEntry{Type: overreachType, DisplayName: "Test Overreach", Description: "Try to write the Router's headers."},
		config.PluginOptions[overreachPlugin]{Strict: true},
	)); err != nil {
		panic(err)
	}
}

func overreachDocument(backend string) string {
	return strings.Replace(stampDocument(backend, ""), "        - type: "+stampType+"\n          configuration:\n",
		"        - type: "+overreachType+"\n          configuration: {}\n", 1)
}

const completion = `{"id":"c","object":"chat.completion","model":"m","choices":[{"index":0,` +
	`"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],` +
	`"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`

func TestAPluginCannotWriteTheRoutersHeadersInEitherMode(t *testing.T) {
	var upstreamSaw http.Header
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.Copy(io.Discard, r.Body)
		upstreamSaw = r.Header.Clone()
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, completion)
	}))
	defer backend.Close()
	router, err := extproc.NewOpenAIRouter(writeDocument(t, overreachDocument(backend.URL)))
	if err != nil {
		t.Fatal(err)
	}
	defer router.Close()
	body := `{"model":"vllm-sr/auto","messages":[{"role":"user","content":"hello there"}]}`
	want := strconv.Itoa(len(refusedHeaders))

	t.Run("extproc", func(t *testing.T) {
		engine := routing.NewEngine(extproc.NewRouterService(router), routing.DefaultOptions)
		header := routing.Header{
			{Name: ":method", Value: http.MethodPost},
			{Name: ":path", Value: "/v1/chat/completions"},
			{Name: ":authority", Value: "router.test"},
			{Name: ":scheme", Value: "http"},
			{Name: "content-type", Value: "application/json"},
		}
		plan, err := engine.Plan(context.Background(), &routing.Request{Header: header, Body: []byte(body)})
		if err != nil || plan.Call == nil {
			t.Fatalf("plan = %+v, %v", plan, err)
		}
		forwarded := plan.Call.Request.Header
		if plan.Call.Route != "m" || forwarded.Get("x-selected-model") != "m" ||
			forwarded.Get("x-envoy-original-path") != "" || forwarded.Get("x-overreach-refused") != want {
			t.Fatalf("Envoy would route %q with %+v", plan.Call.Route, forwarded)
		}
		resp, err := engine.Respond(context.Background(), plan, &routing.UpstreamResponse{
			Status: http.StatusOK, Header: routing.Header{{Name: "content-type", Value: "application/json"}},
			Body: strings.NewReader(completion),
		})
		plan.Finish(err)
		if err != nil || resp.Header.Get("x-vsr-response-path") != "upstream" || resp.Header.Get("x-overreach-response") != "refused" {
			t.Fatalf("client response = %+v, %v", resp, err)
		}
	})

	t.Run("standalone", func(t *testing.T) {
		set, err := upstream.Build(router.Config, upstream.Options{})
		if err != nil {
			t.Fatal(err)
		}
		defer func() {
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			_ = set.Close(ctx)
		}()
		engine := routing.DefaultOptions
		engine.ExecutesFallback = true
		handler, err := gateway.NewHandler(gateway.Options{
			Serving:  gateway.Static(gateway.Serving{Engine: routing.NewEngine(extproc.NewRouterService(router), engine), Upstream: set}),
			Listener: "http-8899",
		})
		if err != nil {
			t.Fatal(err)
		}
		front := httptest.NewServer(handler)
		defer front.Close()
		resp, err := http.Post(front.URL+"/v1/chat/completions", "application/json", strings.NewReader(body))
		if err != nil {
			t.Fatal(err)
		}
		_, _ = io.Copy(io.Discard, resp.Body)
		_ = resp.Body.Close()
		if upstreamSaw.Get("X-Selected-Model") != "m" || upstreamSaw.Get("X-Envoy-Original-Path") != "" ||
			upstreamSaw.Get("X-Overreach-Refused") != want {
			t.Fatalf("the backend saw %v", upstreamSaw)
		}
		if resp.Header.Get("x-vsr-response-path") != "upstream" || resp.Header.Get("x-overreach-response") != "refused" {
			t.Fatalf("the client saw %v", resp.Header)
		}
	})
}
