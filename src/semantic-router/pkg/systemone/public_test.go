package systemone

import (
	"context"
	"encoding/json"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestPublicSystemOneNativeContractAndSeparateGrant(t *testing.T) {
	cfg := &config.RouterConfig{}
	cfg.ModelDeployments = map[string]config.ModelDeployment{"private-key": {Provider: config.ModelRuntimeProvider, Artifact: "vllm-sr/test", ServedName: "private-served"}}
	listener := &config.Listener{APIKeys: []string{"test-key"}, Models: []string{"chat-only"}, SystemOne: &config.ListenerSystemOne{Models: []string{"vllm-sr/test"}}}
	calls := 0
	handler := Handler(cfg, listener, func(_ context.Context, deployment string, body json.RawMessage) (int, []byte, error) {
		calls++
		if deployment != "private-key" {
			t.Fatalf("wrong deployment %q", deployment)
		}
		var native map[string]json.RawMessage
		if json.Unmarshal(body, &native) != nil || string(native["questions"]) != `{"span":{"type":"classification"},"set":{"type":"set"},"choice":{"type":"choice"},"score":{"type":"score"},"noul":{"type":"noul"}}` {
			t.Fatalf("native question order/types changed: %s", body)
		}
		return 200, []byte(`{"model":"private-served","answers":{"choice":{"choice":"yes"}},"spans":{},"sets":{},"usage":{"input_tokens":4},"meta":{"profile":"exact"}}`), nil
	})
	for _, path := range []string{"/v1/systemone", "/v1/decisions"} {
		request := httptest.NewRequest(http.MethodPost, path, strings.NewReader(`{"model":"vllm-sr/test","state":"input","questions":{"span":{"type":"classification"},"set":{"type":"set"},"choice":{"type":"choice"},"score":{"type":"score"},"noul":{"type":"noul"}}}`))
		request.Header.Set("Authorization", "Bearer test-key")
		response := httptest.NewRecorder()
		handler.ServeHTTP(response, request)
		if response.Code != 200 || strings.Contains(response.Body.String(), "private-") || !strings.Contains(response.Body.String(), `"profile":"exact"`) {
			t.Fatalf("native response %d %s", response.Code, response.Body.String())
		}
	}
	if calls != 2 {
		t.Fatalf("calls=%d", calls)
	}
	for _, test := range []struct {
		model, key string
		code       int
	}{{"vllm-sr/test", "", 401}, {"chat-only", "test-key", 403}, {"private-key", "test-key", 403}} {
		request := httptest.NewRequest(http.MethodPost, "/v1/systemone", strings.NewReader(`{"model":"`+test.model+`"}`))
		request.Header.Set("Api-Key", test.key)
		response := httptest.NewRecorder()
		handler.ServeHTTP(response, request)
		if response.Code != test.code {
			t.Fatalf("%+v => %d", test, response.Code)
		}
	}
	if calls != 2 {
		t.Fatal("denied inference reached runtime")
	}
}

func TestPublicDiscoveryAndListenerSelectionFailClosed(t *testing.T) {
	listeners := []config.Listener{{Name: "a", APIKeys: []string{"a"}, SystemOne: &config.ListenerSystemOne{Models: []string{"public-a"}}}, {Name: "b", APIKeys: []string{"b"}, SystemOne: &config.ListenerSystemOne{Models: []string{"public-b"}}}}
	if _, err := SelectListener(listeners, ""); err == nil {
		t.Fatal("must not union listeners")
	}
	selected, err := SelectListener(listeners, "b")
	if err != nil || selected.Name != "b" {
		t.Fatal("explicit selection failed")
	}
	cfg := &config.RouterConfig{}
	cfg.ModelDeployments = map[string]config.ModelDeployment{"private": {Provider: config.ModelRuntimeProvider, Artifact: "/secret/path", PublicName: "public-b"}}
	request := httptest.NewRequest(http.MethodGet, "/v1/systemone/models", nil)
	request.Header.Set("Api-Key", "b")
	response := httptest.NewRecorder()
	Handler(cfg, selected, nil)(response, request)
	if response.Code != 200 || strings.Contains(response.Body.String(), "private") || strings.Contains(response.Body.String(), "secret") || strings.Contains(response.Body.String(), "public-a") {
		t.Fatal(response.Body.String())
	}
	response = httptest.NewRecorder()
	Handler(cfg, nil, nil)(response, request)
	if response.Code != 404 {
		t.Fatal("unpublished listener exposed SystemOne")
	}
}

func TestPublicNativeBoundsAndSanitizesFailures(t *testing.T) {
	cfg := &config.RouterConfig{}
	cfg.ModelDeployments = map[string]config.ModelDeployment{"private": {Provider: config.ModelRuntimeProvider, Artifact: "vllm-sr/test"}}
	listener := &config.Listener{SystemOne: &config.ListenerSystemOne{Models: []string{"vllm-sr/test"}}}
	handler := Handler(cfg, listener, func(context.Context, string, json.RawMessage) (int, []byte, error) {
		return 429, []byte(`{"error":"/private/model/path"}`), io.ErrUnexpectedEOF
	})
	response := httptest.NewRecorder()
	handler(response, httptest.NewRequest(http.MethodPost, "/v1/systemone", strings.NewReader(`{"model":"vllm-sr/test"}`)))
	if response.Code != 429 || strings.Contains(response.Body.String(), "private") {
		t.Fatal(response.Body.String())
	}
	response = httptest.NewRecorder()
	handler(response, httptest.NewRequest(http.MethodPost, "/v1/systemone", strings.NewReader(strings.Repeat("x", (2<<20)+1))))
	if response.Code != 400 {
		t.Fatal("unbounded native request")
	}
}
