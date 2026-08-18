package handlers

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	dashboardauth "github.com/vllm-project/semantic-router/dashboard/backend/auth"
	routerconfig "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestModelVerificationHandlerRunsDirectOpenAIInference(t *testing.T) {
	t.Parallel()

	const apiKey = "model-verification-secret" //nolint:gosec // Canary verifies configured credentials stay private.
	var received map[string]any
	client := modelVerificationRoundTripper(func(r *http.Request) (*http.Response, error) {
		if r.Method != http.MethodPost || r.URL.Path != "/v1/chat/completions" {
			t.Fatalf("provider request = %s %s, want POST /v1/chat/completions", r.Method, r.URL.Path)
		}
		if r.URL.Scheme != "https" || r.URL.Host != "provider.example" {
			t.Fatalf("provider URL = %s, want configured provider host", r.URL)
		}
		if got := r.Header.Get("Authorization"); got != "Bearer "+apiKey {
			t.Fatalf("Authorization = %q", got)
		}
		if got := r.Header.Get("X-Verification-Scope"); got != "configured-backend" {
			t.Fatalf("X-Verification-Scope = %q", got)
		}
		if err := json.NewDecoder(r.Body).Decode(&received); err != nil {
			t.Fatalf("decode provider request: %v", err)
		}
		return modelVerificationHTTPResponse(http.StatusOK, `{"choices":[{"message":{"content":"  OK from provider  "}}]}`), nil
	})

	config := modelVerificationTestConfig(t, "https://provider.example", "openai", apiKey)
	handler := newModelVerificationHandler("active-config.yaml", modelVerificationOptions{
		client: client,
		loadConfig: func(path string) (*routerconfig.RouterConfig, error) {
			if path != "active-config.yaml" {
				t.Fatalf("config path = %q", path)
			}
			return config, nil
		},
		now: func() time.Time { return time.Date(2026, 8, 15, 5, 6, 7, 0, time.UTC) },
	})
	request := httptest.NewRequest(http.MethodPost, modelVerificationPath, strings.NewReader(`{"model":"logical-model"}`))
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, request)

	if response.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", response.Code, response.Body.String())
	}
	if got := response.Header().Get("Cache-Control"); got != "no-store" {
		t.Fatalf("Cache-Control = %q", got)
	}
	if received["model"] != "provider/real-model" || received["stream"] != false {
		t.Fatalf("provider payload = %#v", received)
	}
	if received["max_completion_tokens"] != float64(modelVerificationMaxTokens) {
		t.Fatalf("provider max_completion_tokens = %#v", received["max_completion_tokens"])
	}
	messages, ok := received["messages"].([]any)
	if !ok || len(messages) != 1 || messages[0].(map[string]any)["content"] != modelVerificationPrompt {
		t.Fatalf("provider messages = %#v", received["messages"])
	}

	var payload ModelVerificationResponse
	if err := json.NewDecoder(response.Body).Decode(&payload); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if !payload.Verified || payload.Model != "logical-model" || payload.ProviderModel != "provider/real-model" {
		t.Fatalf("verification response = %#v", payload)
	}
	if payload.Backend != "logical-model/primary" || payload.Provider != "openai" {
		t.Fatalf("verification target = %#v", payload)
	}
	if payload.Summary != "OK from provider" || payload.VerifiedAt != "2026-08-15T05:06:07Z" {
		t.Fatalf("verification evidence = %#v", payload)
	}
	if strings.Contains(response.Body.String(), apiKey) || strings.Contains(response.Body.String(), "provider.example") {
		t.Fatalf("response exposed provider transport details: %s", response.Body.String())
	}
}

func TestModelVerificationHandlerRunsAnthropicInference(t *testing.T) {
	t.Parallel()

	const apiKey = "anthropic-test-secret" //nolint:gosec // Canary verifies configured credentials stay private.
	client := modelVerificationRoundTripper(func(r *http.Request) (*http.Response, error) {
		if r.URL.Path != "/v1/messages" {
			t.Fatalf("path = %q", r.URL.Path)
		}
		if got := r.Header.Get("x-api-key"); got != apiKey {
			t.Fatalf("x-api-key = %q", got)
		}
		if got := r.Header.Get("anthropic-version"); got != "2024-01-01" {
			t.Fatalf("anthropic-version = %q", got)
		}
		var request map[string]any
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			t.Fatal(err)
		}
		if request["model"] != "provider/real-model" || request["stream"] != false {
			t.Fatalf("request = %#v", request)
		}
		return modelVerificationHTTPResponse(http.StatusOK, `{"content":[{"type":"text","text":"OK anthropic"}]}`), nil
	})

	config := modelVerificationTestConfig(t, "https://anthropic.example", "anthropic", apiKey)
	config.ModelConfig["logical-model"] = routerconfig.ModelParams{
		PreferredEndpoints: []string{"logical-model/primary"},
		ExternalModelIDs:   map[string]string{"anthropic": "provider/real-model"},
		APIFormat:          "anthropic",
	}
	config.ProviderProfiles["logical-model/primary"] = routerconfig.ProviderProfile{
		Type:         "anthropic",
		BaseURL:      "https://anthropic.example",
		ExtraHeaders: map[string]string{"anthropic-version": "2024-01-01"},
	}

	handler := newModelVerificationHandler("active-config.yaml", modelVerificationOptions{
		client:     client,
		loadConfig: func(string) (*routerconfig.RouterConfig, error) { return config, nil },
	})
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, httptest.NewRequest(http.MethodPost, modelVerificationPath, strings.NewReader(`{"model":"logical-model"}`)))
	if response.Code != http.StatusOK || !strings.Contains(response.Body.String(), "OK anthropic") {
		t.Fatalf("status = %d, body = %s", response.Code, response.Body.String())
	}
}

func TestModelVerificationHandlerUsesConfiguredLegacyBackendWithoutRouterRouting(t *testing.T) {
	t.Parallel()

	providerCalls := 0
	client := modelVerificationRoundTripper(func(r *http.Request) (*http.Response, error) {
		providerCalls++
		if got := r.URL.String(); got != "https://physical.example:8443/v1/chat/completions" {
			t.Fatalf("direct provider URL = %q", got)
		}
		var payload map[string]any
		if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
			t.Fatalf("decode direct provider request: %v", err)
		}
		if payload["max_tokens"] != float64(modelVerificationMaxTokens) || payload["max_completion_tokens"] != nil {
			t.Fatalf("vLLM token limit payload = %#v", payload)
		}
		return modelVerificationHTTPResponse(http.StatusOK, `{"choices":[{"message":{"content":"direct backend response"}}]}`), nil
	})
	config := &routerconfig.RouterConfig{
		BackendModels: routerconfig.BackendModels{
			ModelConfig: map[string]routerconfig.ModelParams{
				"physical-model": {PreferredEndpoints: []string{"physical-primary"}},
			},
			VLLMEndpoints: []routerconfig.VLLMEndpoint{{
				Name: "physical-primary", Address: "physical.example", Port: 8443, Protocol: "https", Type: "vllm", Weight: 1,
			}},
		},
	}
	handler := newModelVerificationHandler("active-config.yaml", modelVerificationOptions{
		client:     client,
		loadConfig: func(string) (*routerconfig.RouterConfig, error) { return config, nil },
	})
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, httptest.NewRequest(http.MethodPost, modelVerificationPath, strings.NewReader(`{"model":"physical-model"}`)))
	if response.Code != http.StatusOK || providerCalls != 1 {
		t.Fatalf("status = %d, provider calls = %d, body = %s", response.Code, providerCalls, response.Body.String())
	}
}

func TestModelVerificationHandlerSupportsEndpointOwnedPhysicalModel(t *testing.T) {
	t.Parallel()

	client := modelVerificationRoundTripper(func(r *http.Request) (*http.Response, error) {
		if got := r.URL.String(); got != "http://endpoint-owned.example:8000/v1/chat/completions" {
			t.Fatalf("direct provider URL = %q", got)
		}
		var payload map[string]any
		if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
			t.Fatalf("decode provider request: %v", err)
		}
		if payload["model"] != "endpoint-owned-model" {
			t.Fatalf("provider model = %#v", payload["model"])
		}
		return modelVerificationHTTPResponse(http.StatusOK, `{"choices":[{"message":{"content":"OK endpoint owned"}}]}`), nil
	})
	config := &routerconfig.RouterConfig{BackendModels: routerconfig.BackendModels{
		VLLMEndpoints: []routerconfig.VLLMEndpoint{{
			Name: "endpoint-owned-primary", Model: "endpoint-owned-model", Address: "endpoint-owned.example", Port: 8000, Protocol: "http", Type: "vllm", Weight: 1,
		}},
	}}
	handler := newModelVerificationHandler("active-config.yaml", modelVerificationOptions{
		client:     client,
		loadConfig: func(string) (*routerconfig.RouterConfig, error) { return config, nil },
	})
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, httptest.NewRequest(http.MethodPost, modelVerificationPath, strings.NewReader(`{"model":"endpoint-owned-model"}`)))
	if response.Code != http.StatusOK || !strings.Contains(response.Body.String(), "OK endpoint owned") {
		t.Fatalf("status = %d, body = %s", response.Code, response.Body.String())
	}
}

func TestModelVerificationHandlerFailsClosedAndRedactsProviderErrors(t *testing.T) {
	t.Parallel()

	const apiKey = "do-not-expose-this-key" //nolint:gosec // Canary verifies provider errors cannot expose credentials.
	const headerCanary = "extra-header-secret-canary"
	var upstreamErrorBodyRead atomic.Bool
	client := modelVerificationRoundTripper(func(*http.Request) (*http.Response, error) {
		return &http.Response{
			StatusCode: http.StatusUnauthorized,
			Header:     make(http.Header),
			Body: &modelVerificationTrackingBody{
				Reader: strings.NewReader(`{"error":{"message":"credential ` + apiKey + ` and ` + headerCanary + ` were rejected"}}`),
				read:   &upstreamErrorBodyRead,
			},
		}, nil
	})
	config := modelVerificationTestConfig(t, "https://provider.example", "openai", apiKey)
	config.ProviderProfiles["logical-model/primary"] = routerconfig.ProviderProfile{
		Type:    "openai",
		BaseURL: "https://provider.example/v1",
		ExtraHeaders: map[string]string{
			"X-Private-Token": headerCanary,
		},
	}
	handler := newModelVerificationHandler("active-config.yaml", modelVerificationOptions{
		client:     client,
		loadConfig: func(string) (*routerconfig.RouterConfig, error) { return config, nil },
	})
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, httptest.NewRequest(http.MethodPost, modelVerificationPath, strings.NewReader(`{"model":"logical-model"}`)))
	if response.Code != http.StatusBadGateway {
		t.Fatalf("status = %d, body = %s", response.Code, response.Body.String())
	}
	if strings.Contains(response.Body.String(), apiKey) || strings.Contains(response.Body.String(), headerCanary) || strings.Contains(response.Body.String(), "credential") {
		t.Fatalf("provider error text escaped the safe status-only contract: %s", response.Body.String())
	}
	if upstreamErrorBodyRead.Load() {
		t.Fatal("provider error body was read despite the status-only error contract")
	}
}

func TestModelVerificationHandlerRedactsEveryConfiguredHeaderValueFromSuccessEvidence(t *testing.T) {
	t.Parallel()

	const apiKey = "success-echo-api-key"
	const opaqueHeaderCanary = apiKey + "-opaque-header-suffix"
	client := modelVerificationRoundTripper(func(*http.Request) (*http.Response, error) {
		return modelVerificationHTTPResponse(
			http.StatusOK,
			`{"choices":[{"message":{"content":"echo `+apiKey+` and `+opaqueHeaderCanary+`"}}]}`,
		), nil
	})
	config := modelVerificationTestConfig(t, "https://provider.example", "openai", apiKey)
	profile := config.ProviderProfiles["logical-model/primary"]
	profile.ExtraHeaders["X-Provider-Option"] = opaqueHeaderCanary
	config.ProviderProfiles["logical-model/primary"] = profile
	handler := newModelVerificationHandler("active-config.yaml", modelVerificationOptions{
		client:     client,
		loadConfig: func(string) (*routerconfig.RouterConfig, error) { return config, nil },
	})
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, httptest.NewRequest(http.MethodPost, modelVerificationPath, strings.NewReader(`{"model":"logical-model"}`)))
	if response.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", response.Code, response.Body.String())
	}
	if strings.Contains(response.Body.String(), apiKey) ||
		strings.Contains(response.Body.String(), opaqueHeaderCanary) ||
		strings.Contains(response.Body.String(), "opaque-header-suffix") {
		t.Fatalf("generated evidence exposed configured credential material: %s", response.Body.String())
	}
	if !strings.Contains(response.Body.String(), "[REDACTED]") {
		t.Fatalf("generated evidence did not preserve explicit redaction evidence: %s", response.Body.String())
	}
}

func TestModelVerificationHandlerRateLimitsPerAuthenticatedUserAndAudits(t *testing.T) {
	t.Parallel()

	config := &routerconfig.RouterConfig{BackendModels: routerconfig.BackendModels{
		ModelConfig: map[string]routerconfig.ModelParams{"metadata-only": {}},
	}}
	now := time.Date(2026, 8, 15, 5, 6, 7, 0, time.UTC)
	auditor := &modelVerificationTestAuditor{}
	handler := newModelVerificationHandler("active-config.yaml", modelVerificationOptions{
		loadConfig: func(string) (*routerconfig.RouterConfig, error) { return config, nil },
		auditor:    auditor,
		now:        func() time.Time { return now },
	})

	requestForUser := func(userID string) *http.Request {
		request := httptest.NewRequest(http.MethodPost, modelVerificationPath, strings.NewReader(`{"model":"metadata-only"}`))
		return request.WithContext(dashboardauth.WithAuthContext(request.Context(), dashboardauth.AuthContext{UserID: userID}))
	}
	for attempt := 0; attempt < modelVerificationRateLimitCount; attempt++ {
		response := httptest.NewRecorder()
		handler.ServeHTTP(response, requestForUser("user-a"))
		if response.Code != http.StatusUnprocessableEntity {
			t.Fatalf("attempt %d status = %d, body = %s", attempt+1, response.Code, response.Body.String())
		}
	}
	now = now.Add(100 * time.Millisecond)
	rateLimited := httptest.NewRecorder()
	handler.ServeHTTP(rateLimited, requestForUser("user-a"))
	if rateLimited.Code != http.StatusTooManyRequests || rateLimited.Header().Get("Retry-After") != "60" {
		t.Fatalf("rate-limited status = %d, Retry-After = %q, body = %s", rateLimited.Code, rateLimited.Header().Get("Retry-After"), rateLimited.Body.String())
	}

	otherUser := httptest.NewRecorder()
	handler.ServeHTTP(otherUser, requestForUser("user-b"))
	if otherUser.Code != http.StatusUnprocessableEntity {
		t.Fatalf("other user status = %d, body = %s", otherUser.Code, otherUser.Body.String())
	}
	entries := auditor.Entries()
	if len(entries) != modelVerificationRateLimitCount+2 {
		t.Fatalf("audit entries = %d", len(entries))
	}
	lastRateLimited := entries[modelVerificationRateLimitCount]
	if lastRateLimited.UserID != "user-a" || lastRateLimited.Action != "model.inference_verify" || lastRateLimited.StatusCode != http.StatusTooManyRequests {
		t.Fatalf("rate-limit audit = %#v", lastRateLimited)
	}
	if strings.Contains(lastRateLimited.ExtraJSON, "secret") || !strings.Contains(lastRateLimited.ExtraJSON, `"result":"rate_limited"`) {
		t.Fatalf("audit extra = %s", lastRateLimited.ExtraJSON)
	}
}

func TestModelVerificationServiceCoalescesConcurrentRequestsPerModel(t *testing.T) {
	t.Parallel()

	var providerCalls atomic.Int32
	started := make(chan struct{})
	release := make(chan struct{})
	client := modelVerificationRoundTripper(func(*http.Request) (*http.Response, error) {
		if providerCalls.Add(1) == 1 {
			close(started)
		}
		<-release
		return modelVerificationHTTPResponse(http.StatusOK, `{"choices":[{"message":{"content":"OK shared"}}]}`), nil
	})
	config := modelVerificationTestConfig(t, "https://provider.example", "openai", "")
	service := newModelVerificationService("active-config.yaml", modelVerificationOptions{
		client:     client,
		loadConfig: func(string) (*routerconfig.RouterConfig, error) { return config, nil },
	})

	type result struct {
		response ModelVerificationResponse
		err      error
	}
	results := make(chan result, 2)
	go func() {
		response, err := service.Verify(context.Background(), "logical-model")
		results <- result{response: response, err: err}
	}()
	<-started
	go func() {
		response, err := service.Verify(context.Background(), "logical-model")
		results <- result{response: response, err: err}
	}()
	time.Sleep(10 * time.Millisecond)
	close(release)

	for range 2 {
		result := <-results
		if result.err != nil || !result.response.Verified {
			t.Fatalf("shared verification = %#v, %v", result.response, result.err)
		}
	}
	if providerCalls.Load() != 1 {
		t.Fatalf("provider calls = %d, want 1", providerCalls.Load())
	}
}

func TestModelVerificationServiceDoesNotReuseInflightResultAfterBackendChange(t *testing.T) {
	t.Parallel()

	firstConfig := modelVerificationTestConfig(t, "https://provider-v1.example", "openai", "")
	secondConfig := modelVerificationTestConfig(t, "https://provider-v2.example", "openai", "")
	var activeConfig atomic.Pointer[routerconfig.RouterConfig]
	activeConfig.Store(firstConfig)
	firstStarted := make(chan struct{})
	releaseFirst := make(chan struct{})
	released := false
	defer func() {
		if !released {
			close(releaseFirst)
		}
	}()
	var providerCalls atomic.Int32
	client := modelVerificationRoundTripper(func(request *http.Request) (*http.Response, error) {
		providerCalls.Add(1)
		switch request.URL.Host {
		case "provider-v1.example":
			close(firstStarted)
			<-releaseFirst
			return modelVerificationHTTPResponse(http.StatusOK, `{"choices":[{"message":{"content":"OK v1"}}]}`), nil
		case "provider-v2.example":
			return modelVerificationHTTPResponse(http.StatusOK, `{"choices":[{"message":{"content":"OK v2"}}]}`), nil
		default:
			return nil, fmt.Errorf("unexpected provider host")
		}
	})
	service := newModelVerificationService("active-config.yaml", modelVerificationOptions{
		client: client,
		loadConfig: func(string) (*routerconfig.RouterConfig, error) {
			return activeConfig.Load(), nil
		},
	})

	firstResult := make(chan ModelVerificationResponse, 1)
	go func() {
		response, _ := service.Verify(context.Background(), "logical-model")
		firstResult <- response
	}()
	<-firstStarted
	activeConfig.Store(secondConfig)

	secondResult := make(chan ModelVerificationResponse, 1)
	go func() {
		response, _ := service.Verify(context.Background(), "logical-model")
		secondResult <- response
	}()
	select {
	case response := <-secondResult:
		if response.Summary != "OK v2" {
			t.Fatalf("verification after backend change = %#v, want v2 evidence", response)
		}
	case <-time.After(time.Second):
		t.Fatal("verification after backend change reused the blocked v1 request")
	}

	close(releaseFirst)
	released = true
	if response := <-firstResult; response.Summary != "OK v1" {
		t.Fatalf("original verification = %#v, want v1 evidence", response)
	}
	if providerCalls.Load() != 2 {
		t.Fatalf("provider calls = %d, want one per backend generation", providerCalls.Load())
	}
}

func TestModelVerificationServiceDoesNotFollowProviderRedirects(t *testing.T) {
	t.Parallel()

	service := newModelVerificationService("active-config.yaml", modelVerificationOptions{})
	client, ok := service.client.(*http.Client)
	if !ok || client.CheckRedirect == nil {
		t.Fatalf("production verification client = %T, want redirect-refusing *http.Client", service.client)
	}
	if err := client.CheckRedirect(&http.Request{}, []*http.Request{{}}); !errors.Is(err, http.ErrUseLastResponse) {
		t.Fatalf("redirect policy error = %v, want http.ErrUseLastResponse", err)
	}
}

func TestModelVerificationHandlerBoundsRequestsResponsesAndTimeouts(t *testing.T) {
	t.Parallel()

	t.Run("strict request", func(t *testing.T) {
		t.Parallel()
		handler := newModelVerificationHandler("active-config.yaml", modelVerificationOptions{})
		for _, test := range []struct {
			body string
		}{
			{body: `{}`},
			{body: `{"model":"x","extra":true}`},
			{body: `{"model":"x"}{"model":"y"}`},
			{body: `{"model":"` + strings.Repeat("x", maxModelVerificationNameBytes+1) + `"}`},
		} {
			response := httptest.NewRecorder()
			handler.ServeHTTP(response, httptest.NewRequest(http.MethodPost, modelVerificationPath, strings.NewReader(test.body)))
			if response.Code != http.StatusBadRequest {
				t.Fatalf("body %q status = %d, response = %s", test.body, response.Code, response.Body.String())
			}
		}
	})

	t.Run("oversized response", func(t *testing.T) {
		t.Parallel()
		client := modelVerificationRoundTripper(func(*http.Request) (*http.Response, error) {
			return modelVerificationHTTPResponse(http.StatusOK, strings.Repeat("x", modelVerificationMaxResponseBytes+1)), nil
		})
		config := modelVerificationTestConfig(t, "https://provider.example", "openai", "")
		handler := newModelVerificationHandler("active-config.yaml", modelVerificationOptions{
			client:     client,
			loadConfig: func(string) (*routerconfig.RouterConfig, error) { return config, nil },
		})
		response := httptest.NewRecorder()
		handler.ServeHTTP(response, httptest.NewRequest(http.MethodPost, modelVerificationPath, strings.NewReader(`{"model":"logical-model"}`)))
		if response.Code != http.StatusBadGateway || !strings.Contains(response.Body.String(), "response_too_large") {
			t.Fatalf("status = %d, body = %s", response.Code, response.Body.String())
		}
	})

	t.Run("timeout", func(t *testing.T) {
		t.Parallel()
		config := modelVerificationTestConfig(t, "http://provider.invalid", "openai", "")
		handler := newModelVerificationHandler("active-config.yaml", modelVerificationOptions{
			loadConfig: func(string) (*routerconfig.RouterConfig, error) { return config, nil },
			client: modelVerificationRoundTripper(func(request *http.Request) (*http.Response, error) {
				<-request.Context().Done()
				return nil, request.Context().Err()
			}),
			timeout: time.Millisecond,
		})
		response := httptest.NewRecorder()
		handler.ServeHTTP(response, httptest.NewRequest(http.MethodPost, modelVerificationPath, strings.NewReader(`{"model":"logical-model"}`)))
		if response.Code != http.StatusGatewayTimeout || !strings.Contains(response.Body.String(), "verification_timeout") {
			t.Fatalf("status = %d, body = %s", response.Code, response.Body.String())
		}
	})
}

func TestModelVerificationHandlerReportsMissingModelsAndBackends(t *testing.T) {
	t.Parallel()

	config := &routerconfig.RouterConfig{BackendModels: routerconfig.BackendModels{
		ModelConfig: map[string]routerconfig.ModelParams{"metadata-only": {}},
	}}
	handler := newModelVerificationHandler("active-config.yaml", modelVerificationOptions{
		loadConfig: func(string) (*routerconfig.RouterConfig, error) { return config, nil },
	})
	for _, test := range []struct {
		model  string
		status int
		code   string
	}{
		{model: "missing", status: http.StatusNotFound, code: "model_not_found"},
		{model: "metadata-only", status: http.StatusUnprocessableEntity, code: "backend_unavailable"},
	} {
		response := httptest.NewRecorder()
		handler.ServeHTTP(response, httptest.NewRequest(http.MethodPost, modelVerificationPath, strings.NewReader(`{"model":"`+test.model+`"}`)))
		if response.Code != test.status || !strings.Contains(response.Body.String(), test.code) {
			t.Fatalf("model %q status = %d, body = %s", test.model, response.Code, response.Body.String())
		}
	}
}

func TestModelVerificationHandlerCannotBeUsedAsAnArbitraryProviderProxy(t *testing.T) {
	t.Parallel()

	var outboundCalls atomic.Int32
	client := modelVerificationRoundTripper(func(*http.Request) (*http.Response, error) {
		outboundCalls.Add(1)
		return nil, errors.New("unexpected outbound request")
	})
	run := func(t *testing.T, config *routerconfig.RouterConfig, model string) *httptest.ResponseRecorder {
		t.Helper()
		handler := newModelVerificationHandler("active-config.yaml", modelVerificationOptions{
			client:     client,
			loadConfig: func(string) (*routerconfig.RouterConfig, error) { return config, nil },
		})
		response := httptest.NewRecorder()
		handler.ServeHTTP(
			response,
			httptest.NewRequest(http.MethodPost, modelVerificationPath, strings.NewReader(`{"model":"`+model+`"}`)),
		)
		return response
	}

	t.Run("caller URL is only an exact model lookup", func(t *testing.T) {
		config := modelVerificationTestConfig(t, "https://provider.example", "openai", "")
		response := run(t, config, "https://attacker.invalid/v1/chat/completions")
		if response.Code != http.StatusNotFound {
			t.Fatalf("status = %d, body = %s", response.Code, response.Body.String())
		}
	})

	t.Run("provider base URL cannot contain user info", func(t *testing.T) {
		config := modelVerificationTestConfig(t, "https://provider.example", "openai", "")
		profile := config.ProviderProfiles["logical-model/primary"]
		profile.BaseURL = "https://credential@provider.example/v1"
		config.ProviderProfiles["logical-model/primary"] = profile
		response := run(t, config, "logical-model")
		if response.Code != http.StatusUnprocessableEntity || !strings.Contains(response.Body.String(), "backend_invalid") {
			t.Fatalf("status = %d, body = %s", response.Code, response.Body.String())
		}
	})

	t.Run("provider chat path cannot replace configured host", func(t *testing.T) {
		config := modelVerificationTestConfig(t, "https://provider.example", "openai", "")
		profile := config.ProviderProfiles["logical-model/primary"]
		profile.ChatPath = "//attacker.invalid/v1/chat/completions"
		config.ProviderProfiles["logical-model/primary"] = profile
		response := run(t, config, "logical-model")
		if response.Code != http.StatusUnprocessableEntity || !strings.Contains(response.Body.String(), "backend_invalid") {
			t.Fatalf("status = %d, body = %s", response.Code, response.Body.String())
		}
	})

	t.Run("legacy address must be a host, not an authority with user info", func(t *testing.T) {
		config := &routerconfig.RouterConfig{BackendModels: routerconfig.BackendModels{
			ModelConfig: map[string]routerconfig.ModelParams{
				"logical-model": {PreferredEndpoints: []string{"physical-primary"}},
			},
			VLLMEndpoints: []routerconfig.VLLMEndpoint{{
				Name: "physical-primary", Address: "credential@provider.example", Protocol: "https", Type: "vllm", Weight: 1,
			}},
		}}
		response := run(t, config, "logical-model")
		if response.Code != http.StatusUnprocessableEntity || !strings.Contains(response.Body.String(), "backend_invalid") {
			t.Fatalf("status = %d, body = %s", response.Code, response.Body.String())
		}
	})

	if outboundCalls.Load() != 0 {
		t.Fatalf("untrusted target metadata caused %d outbound requests", outboundCalls.Load())
	}
}

func TestModelVerificationHandlerRejectsOtherMethods(t *testing.T) {
	t.Parallel()
	auditor := &modelVerificationTestAuditor{}
	handler := newModelVerificationHandler("active-config.yaml", modelVerificationOptions{auditor: auditor})
	response := httptest.NewRecorder()
	request := httptest.NewRequest(http.MethodGet, modelVerificationPath, nil)
	cancelledContext, cancel := context.WithCancel(request.Context())
	cancel()
	handler.ServeHTTP(response, request.WithContext(cancelledContext))
	if response.Code != http.StatusMethodNotAllowed || response.Header().Get("Allow") != http.MethodPost {
		t.Fatalf("status = %d, Allow = %q", response.Code, response.Header().Get("Allow"))
	}
	if response.Header().Get("Cache-Control") != "no-store" {
		t.Fatalf("Cache-Control = %q", response.Header().Get("Cache-Control"))
	}
	if errors := auditor.ContextErrors(); len(errors) != 1 || errors[0] != nil {
		t.Fatalf("durable audit contexts = %#v, want one context detached from request cancellation", errors)
	}
}

func modelVerificationTestConfig(t *testing.T, baseURL, providerType, apiKey string) *routerconfig.RouterConfig {
	t.Helper()
	if _, err := url.ParseRequestURI(baseURL); err != nil {
		t.Fatalf("invalid test base URL: %v", err)
	}
	const endpointName = "logical-model/primary"
	return &routerconfig.RouterConfig{
		BackendModels: routerconfig.BackendModels{
			ModelConfig: map[string]routerconfig.ModelParams{
				"logical-model": {
					PreferredEndpoints: []string{endpointName},
					ExternalModelIDs:   map[string]string{providerType: "provider/real-model"},
					APIFormat:          providerType,
				},
			},
			VLLMEndpoints: []routerconfig.VLLMEndpoint{{
				Name: endpointName, Type: providerType, Weight: 10, APIKey: apiKey, ProviderProfileName: endpointName,
			}},
			ProviderProfiles: map[string]routerconfig.ProviderProfile{
				endpointName: {
					Type:         providerType,
					BaseURL:      baseURL + "/v1",
					ExtraHeaders: map[string]string{"X-Verification-Scope": "configured-backend"},
				},
			},
		},
	}
}

type modelVerificationRoundTripper func(*http.Request) (*http.Response, error)

func (fn modelVerificationRoundTripper) Do(request *http.Request) (*http.Response, error) {
	return fn(request)
}

type modelVerificationTrackingBody struct {
	io.Reader
	read *atomic.Bool
}

func (body *modelVerificationTrackingBody) Read(buffer []byte) (int, error) {
	body.read.Store(true)
	return body.Reader.Read(buffer)
}

func (*modelVerificationTrackingBody) Close() error { return nil }

func modelVerificationHTTPResponse(status int, body string) *http.Response {
	return &http.Response{
		StatusCode: status,
		Header:     make(http.Header),
		Body:       io.NopCloser(strings.NewReader(body)),
	}
}

type modelVerificationTestAuditor struct {
	mu            sync.Mutex
	entries       []dashboardauth.AuditLog
	contextErrors []error
}

func (auditor *modelVerificationTestAuditor) AddAuditLog(ctx context.Context, entry dashboardauth.AuditLog) error {
	auditor.mu.Lock()
	defer auditor.mu.Unlock()
	auditor.entries = append(auditor.entries, entry)
	auditor.contextErrors = append(auditor.contextErrors, ctx.Err())
	return nil
}

func (auditor *modelVerificationTestAuditor) Entries() []dashboardauth.AuditLog {
	auditor.mu.Lock()
	defer auditor.mu.Unlock()
	return append([]dashboardauth.AuditLog(nil), auditor.entries...)
}

func (auditor *modelVerificationTestAuditor) ContextErrors() []error {
	auditor.mu.Lock()
	defer auditor.mu.Unlock()
	return append([]error(nil), auditor.contextErrors...)
}

func TestModelVerificationServiceMasksConfigLoaderFailures(t *testing.T) {
	t.Parallel()
	service := newModelVerificationService("active-config.yaml", modelVerificationOptions{
		loadConfig: func(string) (*routerconfig.RouterConfig, error) {
			return nil, errors.New("secret config path and value")
		},
	})
	_, err := service.Verify(context.Background(), "model")
	var verificationErr *modelVerificationError
	if !errors.As(err, &verificationErr) || verificationErr.code != "config_unavailable" {
		t.Fatalf("error = %#v", err)
	}
	if strings.Contains(err.Error(), "secret") {
		t.Fatalf("config loader error leaked: %v", err)
	}
}
