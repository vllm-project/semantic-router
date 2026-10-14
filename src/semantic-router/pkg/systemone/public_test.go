package systemone

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
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

func TestPublicSystemOneRequiresNonemptyCredentialsOnProtectedListeners(t *testing.T) {
	cfg := &config.RouterConfig{}
	cfg.ModelDeployments = map[string]config.ModelDeployment{"private": {Provider: config.ModelRuntimeProvider, Artifact: "vllm-sr/test"}}
	for _, test := range []struct {
		name          string
		keys          []string
		authorization string
		apiKey        string
		wantStatus    int
	}{
		{name: "open listener", wantStatus: http.StatusOK},
		{name: "missing credentials", keys: []string{"test-key"}, wantStatus: http.StatusUnauthorized},
		{name: "empty configured key", keys: []string{""}, wantStatus: http.StatusUnauthorized},
		{name: "mixed keys without credentials", keys: []string{"", "test-key"}, wantStatus: http.StatusUnauthorized},
		{name: "empty bearer", keys: []string{"", "test-key"}, authorization: "Bearer \t", wantStatus: http.StatusUnauthorized},
		{name: "malformed bearer", keys: []string{"", "test-key"}, authorization: "Bearertest-key", wantStatus: http.StatusUnauthorized},
		{name: "wrong bearer", keys: []string{"", "test-key"}, authorization: "Bearer wrong-key", wantStatus: http.StatusUnauthorized},
		{name: "wrong api key", keys: []string{"", "test-key"}, apiKey: "wrong-key", wantStatus: http.StatusUnauthorized},
		{name: "valid bearer", keys: []string{"", "test-key"}, authorization: "Bearer test-key", wantStatus: http.StatusOK},
		{name: "valid api key", keys: []string{"", "test-key"}, apiKey: "test-key", wantStatus: http.StatusOK},
		{name: "api key fallback", keys: []string{"", "test-key"}, authorization: "Bearer wrong-key", apiKey: "test-key", wantStatus: http.StatusOK},
	} {
		t.Run(test.name, func(t *testing.T) {
			listener := &config.Listener{APIKeys: test.keys, SystemOne: &config.ListenerSystemOne{Models: []string{"vllm-sr/test"}}}
			for _, path := range []string{"/v1/systemone", "/v1/systemone/models"} {
				t.Run(path, func(t *testing.T) {
					calls := 0
					handler := Handler(cfg, listener, func(context.Context, string, json.RawMessage) (int, []byte, error) {
						calls++
						return http.StatusOK, []byte(`{"answers":{}}`), nil
					})
					method := http.MethodPost
					if path == "/v1/systemone/models" {
						method = http.MethodGet
					}
					request := httptest.NewRequest(method, path, strings.NewReader(`{"model":"vllm-sr/test"}`))
					request.Header.Set("Authorization", test.authorization)
					request.Header.Set("Api-Key", test.apiKey)
					response := httptest.NewRecorder()
					handler(response, request)
					if response.Code != test.wantStatus {
						t.Fatalf("status=%d, want %d: %s", response.Code, test.wantStatus, response.Body.String())
					}
					if test.wantStatus == http.StatusUnauthorized {
						if calls != 0 || !strings.Contains(response.Body.String(), "invalid_api_key") || response.Header().Get("WWW-Authenticate") == "" {
							t.Fatalf("denied request reached runtime or omitted auth challenge: calls=%d response=%s", calls, response.Body.String())
						}
					}
				})
			}
		})
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
	handler(response, httptest.NewRequest(http.MethodPost, "/v1/systemone", strings.NewReader(strings.Repeat("x", RequestLimit+1))))
	if response.Code != http.StatusRequestEntityTooLarge || !strings.Contains(response.Body.String(), `"code":"request_too_large"`) {
		t.Fatalf("over-limit native request: %d %s", response.Code, response.Body.String())
	}
}

func TestPublicNativeRuntimeRejectionsKeepTheirCodeAndReason(t *testing.T) {
	cfg := &config.RouterConfig{}
	cfg.ModelDeployments = map[string]config.ModelDeployment{"private": {Provider: config.ModelRuntimeProvider, Artifact: "vllm-sr/test"}}
	listener := &config.Listener{SystemOne: &config.ListenerSystemOne{Models: []string{"vllm-sr/test"}}}
	rejection := func(code, reason string, limit int) string {
		body := `{"error":{"code":"` + code + `","message":"videos[0]: model /private/models/test failed"`
		if reason != "" {
			body += `,"reason":"` + reason + `"`
		}
		if limit != 0 {
			body += `,"limit":` + strconv.Itoa(limit)
		}
		return body + `}}`
	}
	for _, test := range []struct {
		status      int
		body        string
		wantStatus  int
		wantCode    string
		wantMessage string
	}{
		{400, rejection("invalid_request", "video_too_large", 32_000_000), 400, "invalid_request", "video exceeds 32 MB"},
		{400, rejection("invalid_request", "video_too_long", 300), 400, "invalid_request", "video longer than 300 s"},
		{400, rejection("invalid_request", "video_malformed", 0), 400, "invalid_request", "malformed video"},
		{400, rejection("invalid_request", "video_pixels", 8_294_400), 400, "invalid_request", "video frames exceed 8,294,400 pixels"},
		{400, rejection("invalid_request", "video_tokens", 16_384), 400, "invalid_request", "videos exceed 16,384 input tokens"},
		{400, rejection("invalid_request", "image_too_large", 8_000_000), 400, "invalid_request", "image exceeds 8 MB"},
		{400, rejection("invalid_request", "image_too_large", 1_500_000_001), 400, "invalid_request", "image exceeds 1,500,000,001 bytes"},
		{400, rejection("invalid_request", "image_unsupported", 0), 400, "invalid_request", "model does not read images"},
		{400, rejection("invalid_request", "video_too_long", 0), 400, "invalid_request", "invalid request"},
		{400, rejection("invalid_request", "", 0), 400, "invalid_request", "invalid request"},
		{400, rejection("invalid_request", "/private/reason", 0), 400, "invalid_request", "invalid request"},
		{404, rejection("model_not_found", "", 0), 404, "model_not_found", "model not found"},
		{413, rejection("request_too_large", "", 0), 413, "request_too_large", "request too large"},
		{503, rejection("not_ready", "", 0), 503, "not_ready", "not ready"},
		{500, rejection("internal_error", "", 0), 500, "systemone_request_failed", "systemone request failed"},
		{502, `not json /private/models/test`, 502, "systemone_request_failed", "systemone request failed"},
	} {
		handler := Handler(cfg, listener, func(context.Context, string, json.RawMessage) (int, []byte, error) {
			return test.status, []byte(test.body), nil
		})
		response := httptest.NewRecorder()
		handler(response, httptest.NewRequest(http.MethodPost, "/v1/systemone", strings.NewReader(`{"model":"vllm-sr/test"}`)))
		var got struct {
			Error struct{ Code, Message string } `json:"error"`
		}
		if json.Unmarshal(response.Body.Bytes(), &got) != nil || response.Code != test.wantStatus ||
			got.Error.Code != test.wantCode || got.Error.Message != test.wantMessage || strings.Contains(response.Body.String(), "private") {
			t.Fatalf("runtime %d %s => %d %s", test.status, test.body, response.Code, response.Body.String())
		}
	}
}

func TestPublicNativeRefusalsTellClientsWhenToRetry(t *testing.T) {
	cfg := &config.RouterConfig{}
	cfg.ModelDeployments = map[string]config.ModelDeployment{"private": {Provider: config.ModelRuntimeProvider, Artifact: "vllm-sr/test"}}
	listener := &config.Listener{SystemOne: &config.ListenerSystemOne{Models: []string{"vllm-sr/test"}}}
	for _, test := range []struct {
		name       string
		status     int
		body       string
		err        error
		wantStatus int
		wantCode   string
		wantRetry  string
	}{
		{"worker overloaded", 429, `{"error":{"code":"overloaded","message":"busy"}}`, nil, 429, "overloaded", "1"},
		{"pool overloaded", 429, `{"error":{"code":"overloaded","message":"busy"}}`, io.ErrUnexpectedEOF, 429, "overloaded", "1"},
		{"model not ready", 503, `{"error":{"code":"not_ready","message":"loading"}}`, nil, 503, "not_ready", "5"},
		{"transport failure", 0, ``, io.ErrUnexpectedEOF, 503, "systemone_unavailable", "5"},
		{"engine deadline", 0, ``, context.DeadlineExceeded, 504, "systemone_deadline_exceeded", ""},
		{"rejected request", 400, `{"error":{"code":"invalid_request","message":"bad"}}`, nil, 400, "invalid_request", ""},
	} {
		t.Run(test.name, func(t *testing.T) {
			handler := Handler(cfg, listener, func(context.Context, string, json.RawMessage) (int, []byte, error) {
				return test.status, []byte(test.body), test.err
			})
			response := httptest.NewRecorder()
			handler(response, httptest.NewRequest(http.MethodPost, "/v1/systemone", strings.NewReader(`{"model":"vllm-sr/test"}`)))
			var got struct {
				Error struct{ Code string } `json:"error"`
			}
			if json.Unmarshal(response.Body.Bytes(), &got) != nil || response.Code != test.wantStatus || got.Error.Code != test.wantCode {
				t.Fatalf("got %d %s", response.Code, response.Body.String())
			}
			if retry := response.Header().Get("Retry-After"); retry != test.wantRetry {
				t.Fatalf("Retry-After %q, want %q", retry, test.wantRetry)
			}
		})
	}
}
