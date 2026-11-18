package handlers

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	routerconfig "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

const modelHealthTestKey = "model-health-secret" //nolint:gosec // Canary verifies configured credentials stay private.

type modelHealthTestClock struct {
	mu  sync.Mutex
	now time.Time
}

func (clock *modelHealthTestClock) Now() time.Time {
	clock.mu.Lock()
	defer clock.mu.Unlock()
	return clock.now
}

func (clock *modelHealthTestClock) Advance(duration time.Duration) {
	clock.mu.Lock()
	defer clock.mu.Unlock()
	clock.now = clock.now.Add(duration)
}

func newModelHealthTestService(config *routerconfig.RouterConfig, client modelVerificationHTTPClient, clock *modelHealthTestClock) *modelHealthService {
	options := modelHealthOptions{
		client:     client,
		loadConfig: func(string) (*routerconfig.RouterConfig, error) { return config, nil },
		timeout:    time.Second,
	}
	if clock != nil {
		options.now = clock.Now
	}
	return newModelHealthService("active-config.yaml", options)
}

func checkModelHealth(t *testing.T, service *modelHealthService, refresh bool, models ...string) ModelHealthResponse {
	t.Helper()
	response, err := service.Check(context.Background(), models, refresh)
	if err != nil {
		t.Fatalf("Check() error = %v", err)
	}
	return response
}

func onlyModelHealth(t *testing.T, response ModelHealthResponse) ModelHealthEntry {
	t.Helper()
	if len(response.Models) != 1 {
		t.Fatalf("models = %+v, want one entry", response.Models)
	}
	return response.Models[0]
}

func TestModelHealthChecksTheModelListWithConfiguredCredentials(t *testing.T) {
	t.Parallel()

	client := modelVerificationRoundTripper(func(r *http.Request) (*http.Response, error) {
		if r.Method != http.MethodGet || r.URL.String() != "https://provider.example/v1/models" {
			t.Errorf("health request = %s %s, want GET https://provider.example/v1/models", r.Method, r.URL)
		}
		if got := r.Header.Get("Authorization"); got != "Bearer "+modelHealthTestKey {
			t.Errorf("Authorization = %q", got)
		}
		if got := r.Header.Get("X-Verification-Scope"); got != "configured-backend" {
			t.Errorf("X-Verification-Scope = %q", got)
		}
		return modelVerificationHTTPResponse(http.StatusOK, `{"data":[{"id":"other"},{"id":"provider/real-model"}]}`), nil
	})
	config := modelVerificationTestConfig(t, "https://provider.example", "openai", modelHealthTestKey)
	response := checkModelHealth(t, newModelHealthTestService(config, client, nil), false)

	entry := onlyModelHealth(t, response)
	if entry.Model != "logical-model" || entry.Status != ModelHealthStatusHealthy || entry.Reason != "" {
		t.Fatalf("entry = %+v, want a healthy logical-model", entry)
	}
	if entry.LatencyMS == nil || entry.P95MS == nil || len(entry.RecentMS) != 1 || entry.CheckedAt == "" {
		t.Fatalf("entry latency = %+v, want latest, p95 and one recent sample", entry)
	}
	if len(entry.Endpoints) != 1 || entry.Endpoints[0].Name != "logical-model/primary" || entry.Endpoints[0].HTTPStatus != http.StatusOK {
		t.Fatalf("endpoints = %+v", entry.Endpoints)
	}
	if response.IntervalSeconds != 30 {
		t.Fatalf("intervalSeconds = %d, want 30", response.IntervalSeconds)
	}
	encoded, _ := json.Marshal(response)
	for _, private := range []string{modelHealthTestKey, "provider.example", "configured-backend"} {
		if strings.Contains(string(encoded), private) {
			t.Fatalf("response leaks %q: %s", private, encoded)
		}
	}
}

func TestModelHealthDegradesWhenTheBackendDoesNotListTheModel(t *testing.T) {
	t.Parallel()

	for _, test := range []struct {
		name   string
		body   string
		status string
	}{
		{name: "complete list without the model", body: `{"data":[{"id":"other"}]}`, status: ModelHealthStatusDegraded},
		{name: "one page of many", body: `{"data":[{"id":"other"}],"has_more":true}`, status: ModelHealthStatusHealthy},
		{name: "unreadable list", body: `not json`, status: ModelHealthStatusHealthy},
		{name: "models key with prefix", body: `{"models":[{"name":"models/provider/real-model"}]}`, status: ModelHealthStatusHealthy},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			client := modelVerificationRoundTripper(func(*http.Request) (*http.Response, error) {
				return modelVerificationHTTPResponse(http.StatusOK, test.body), nil
			})
			config := modelVerificationTestConfig(t, "https://provider.example", "openai", modelHealthTestKey)
			entry := onlyModelHealth(t, checkModelHealth(t, newModelHealthTestService(config, client, nil), false))
			if entry.Status != test.status {
				t.Fatalf("status = %q (%s), want %q", entry.Status, entry.Reason, test.status)
			}
			if test.status == ModelHealthStatusDegraded && entry.Reason != "The backend does not list provider/real-model." {
				t.Fatalf("reason = %q", entry.Reason)
			}
		})
	}
}

func TestModelHealthUsesTheConfiguredHealthPathWithoutCredentials(t *testing.T) {
	t.Parallel()

	client := modelVerificationRoundTripper(func(r *http.Request) (*http.Response, error) {
		if r.URL.String() != "https://provider.example/healthz?probe=1" {
			t.Errorf("health request URL = %s", r.URL)
		}
		if r.Header.Get("Authorization") != "" || r.Header.Get("X-Verification-Scope") != "" {
			t.Errorf("health path request carried backend headers: %v", r.Header)
		}
		return modelVerificationHTTPResponse(http.StatusServiceUnavailable, ""), nil
	})
	config := modelVerificationTestConfig(t, "https://provider.example", "openai", modelHealthTestKey)
	params := config.ModelConfig["logical-model"]
	params.Reliability.HealthCheckPath = "/healthz?probe=1"
	config.ModelConfig["logical-model"] = params

	entry := onlyModelHealth(t, checkModelHealth(t, newModelHealthTestService(config, client, nil), false))
	if entry.Status != ModelHealthStatusDown || entry.Reason != "The health check answered HTTP 503." {
		t.Fatalf("entry = %+v, want down from the health path", entry)
	}
	if entry.LatencyMS != nil || entry.P95MS != nil {
		t.Fatalf("a failed check reported latency: %+v", entry)
	}
}

func TestModelHealthClassifiesAnswers(t *testing.T) {
	t.Parallel()

	list := modelHealthTarget{check: modelHealthCheckList}
	for _, test := range []struct {
		name    string
		target  modelHealthTarget
		status  int
		latency time.Duration
		want    string
		reason  string
	}{
		{name: "ok", target: list, status: http.StatusOK, want: ModelHealthStatusHealthy},
		{name: "slow", target: list, status: http.StatusOK, latency: 3200 * time.Millisecond, want: ModelHealthStatusDegraded, reason: "Slow: answered in 3.2 s."},
		{name: "credentials rejected", target: list, status: http.StatusUnauthorized, want: ModelHealthStatusDown, reason: "The backend rejected its credentials (HTTP 401)."},
		{name: "caller credential", target: modelHealthTarget{check: modelHealthCheckList, callerAuth: true}, status: http.StatusForbidden, want: ModelHealthStatusHealthy, reason: "Reachable; requests carry the caller's credential."},
		{name: "rate limited", target: list, status: http.StatusTooManyRequests, want: ModelHealthStatusDegraded, reason: "The backend is rate limiting requests (HTTP 429)."},
		{name: "server error", target: list, status: http.StatusBadGateway, want: ModelHealthStatusDown, reason: "The backend answered HTTP 502."},
		{name: "no model list", target: list, status: http.StatusNotFound, want: ModelHealthStatusDegraded, reason: "The model list answered HTTP 404."},
		{name: "reachable without a list", target: modelHealthTarget{check: modelHealthCheckReach}, status: http.StatusNotFound, want: ModelHealthStatusHealthy, reason: "Reachable; this provider has no model list to confirm the model."},
		{name: "health path redirect", target: modelHealthTarget{check: modelHealthCheckPath}, status: http.StatusFound, want: ModelHealthStatusDown, reason: "The health check answered HTTP 302."},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			var result modelHealthProbe
			classifyModelHealthAnswer(&result, test.target, test.status, strings.NewReader(""), test.latency)
			if result.status != test.want || result.reason != test.reason {
				t.Fatalf("classified %d as %q (%q), want %q (%q)", test.status, result.status, result.reason, test.want, test.reason)
			}
			if result.answered != (test.want != ModelHealthStatusDown) {
				t.Fatalf("answered = %v for %q", result.answered, test.want)
			}
		})
	}
}

func TestModelHealthReportsUnreachableAndSilentBackends(t *testing.T) {
	t.Parallel()

	for _, test := range []struct {
		name   string
		client modelVerificationRoundTripper
		reason string
	}{
		{
			name: "connection refused",
			client: func(*http.Request) (*http.Response, error) {
				return nil, errors.New("dial tcp 10.1.2.3:443: connect: connection refused")
			},
			reason: "The backend could not be reached.",
		},
		{
			name: "no answer",
			client: func(r *http.Request) (*http.Response, error) {
				<-r.Context().Done()
				return nil, r.Context().Err()
			},
			reason: "No answer within 30 ms.",
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			config := modelVerificationTestConfig(t, "https://provider.example", "openai", modelHealthTestKey)
			service := newModelHealthService("active-config.yaml", modelHealthOptions{
				client:     test.client,
				loadConfig: func(string) (*routerconfig.RouterConfig, error) { return config, nil },
				timeout:    30 * time.Millisecond,
			})
			entry := onlyModelHealth(t, checkModelHealth(t, service, false))
			if entry.Status != ModelHealthStatusDown || entry.Reason != test.reason {
				t.Fatalf("entry = %+v, want down: %s", entry, test.reason)
			}
			if strings.Contains(entry.Reason, "10.1.2.3") {
				t.Fatalf("reason leaks the backend address: %q", entry.Reason)
			}
		})
	}
}

func TestModelHealthAggregatesEveryBackendOfAModel(t *testing.T) {
	t.Parallel()

	config := modelVerificationTestConfig(t, "https://provider.example", "openai", modelHealthTestKey)
	config.VLLMEndpoints = append(config.VLLMEndpoints, routerconfig.VLLMEndpoint{
		Name: "logical-model_replica", Type: "openai", Weight: 5, APIKey: modelHealthTestKey, ProviderProfileName: "logical-model_replica",
	})
	config.ProviderProfiles["logical-model_replica"] = routerconfig.ProviderProfile{Type: "openai", BaseURL: "https://replica.example/v1"}
	params := config.ModelConfig["logical-model"]
	params.PreferredEndpoints = append(params.PreferredEndpoints, "logical-model_replica")
	config.ModelConfig["logical-model"] = params

	client := modelVerificationRoundTripper(func(r *http.Request) (*http.Response, error) {
		if r.URL.Host == "replica.example" {
			return modelVerificationHTTPResponse(http.StatusServiceUnavailable, ""), nil
		}
		return modelVerificationHTTPResponse(http.StatusOK, `{"data":[{"id":"provider/real-model"}]}`), nil
	})
	entry := onlyModelHealth(t, checkModelHealth(t, newModelHealthTestService(config, client, nil), false))
	if entry.Status != ModelHealthStatusDegraded || entry.Reason != "Down on 1 of 2 backends." {
		t.Fatalf("entry = %+v, want degraded with one backend down", entry)
	}
	if len(entry.Endpoints) != 2 || entry.Endpoints[1].Name != "replica" || entry.Endpoints[1].Status != ModelHealthStatusDown {
		t.Fatalf("endpoints = %+v, want the replica named as its backend ref", entry.Endpoints)
	}
	if entry.LatencyMS == nil || len(entry.RecentMS) != 1 {
		t.Fatalf("latency = %+v, want the answering backend only", entry)
	}
}

func TestModelHealthExplainsModelsWithNothingToCheck(t *testing.T) {
	t.Parallel()

	config := &routerconfig.RouterConfig{
		BackendModels: routerconfig.BackendModels{
			ModelConfig: map[string]routerconfig.ModelParams{
				"catalog-only": {},
				"runtime-alias": {
					Deployment: "decision-kai",
				},
			},
		},
	}
	client := modelVerificationRoundTripper(func(r *http.Request) (*http.Response, error) {
		t.Errorf("unexpected health request %s", r.URL)
		return nil, errors.New("unexpected request")
	})
	response := checkModelHealth(t, newModelHealthTestService(config, client, nil), false)
	reasons := map[string]string{}
	for _, entry := range response.Models {
		if entry.Status != ModelHealthStatusUnchecked {
			t.Fatalf("entry = %+v, want unchecked", entry)
		}
		reasons[entry.Model] = entry.Reason
	}
	if reasons["catalog-only"] != "No backend endpoint to check." || reasons["runtime-alias"] != "Served by the shared deployment decision-kai." {
		t.Fatalf("reasons = %+v", reasons)
	}
}

func TestModelHealthLeavesBackendsWithUnreadableTLSFilesUnchecked(t *testing.T) {
	t.Parallel()

	config := modelVerificationTestConfig(t, "https://provider.example", "openai", modelHealthTestKey)
	profile := config.ProviderProfiles["logical-model/primary"]
	profile.TLS = &routerconfig.DeploymentTLS{CAFile: t.TempDir() + "/missing-ca.pem"}
	config.ProviderProfiles["logical-model/primary"] = profile
	client := modelVerificationRoundTripper(func(r *http.Request) (*http.Response, error) {
		t.Errorf("unexpected health request %s", r.URL)
		return nil, errors.New("unexpected request")
	})
	entry := onlyModelHealth(t, checkModelHealth(t, newModelHealthTestService(config, client, nil), false))
	if entry.Status != ModelHealthStatusUnchecked || entry.Reason != "The Dashboard cannot read this backend's TLS files." {
		t.Fatalf("entry = %+v", entry)
	}
	if strings.Contains(entry.Reason, "missing-ca.pem") {
		t.Fatalf("reason leaks the file path: %q", entry.Reason)
	}
}

func TestModelHealthSharesFreshChecksAndHonorsTheRefreshFloor(t *testing.T) {
	t.Parallel()

	var requests atomic.Int32
	client := modelVerificationRoundTripper(func(*http.Request) (*http.Response, error) {
		requests.Add(1)
		return modelVerificationHTTPResponse(http.StatusOK, `{"data":[{"id":"provider/real-model"}]}`), nil
	})
	clock := &modelHealthTestClock{now: time.Date(2026, 11, 18, 9, 0, 0, 0, time.UTC)}
	config := modelVerificationTestConfig(t, "https://provider.example", "openai", modelHealthTestKey)
	service := newModelHealthTestService(config, client, clock)

	steps := []struct {
		advance  time.Duration
		refresh  bool
		requests int32
	}{
		{requests: 1},
		{advance: 10 * time.Second, requests: 1},
		{advance: time.Second, refresh: true, requests: 2},
		{advance: 2 * time.Second, refresh: true, requests: 2},
		{advance: 25 * time.Second, requests: 3},
	}
	var last ModelHealthEntry
	for index, step := range steps {
		clock.Advance(step.advance)
		last = onlyModelHealth(t, checkModelHealth(t, service, step.refresh, "logical-model"))
		if got := requests.Load(); got != step.requests {
			t.Fatalf("step %d: requests = %d, want %d", index, got, step.requests)
		}
	}
	if len(last.RecentMS) != 3 {
		t.Fatalf("recent samples = %v, want one per check", last.RecentMS)
	}

	params := config.ModelConfig["logical-model"]
	params.ExternalModelIDs = map[string]string{"openai": "provider/renamed-model"}
	config.ModelConfig["logical-model"] = params
	changed := onlyModelHealth(t, checkModelHealth(t, service, false))
	if requests.Load() != 4 || len(changed.RecentMS) != 1 {
		t.Fatalf("after a config change: requests = %d, samples = %v; want a new check and fresh history", requests.Load(), changed.RecentMS)
	}
}

func TestModelHealthHandler(t *testing.T) {
	t.Parallel()

	client := modelVerificationRoundTripper(func(*http.Request) (*http.Response, error) {
		return modelVerificationHTTPResponse(http.StatusOK, `{"data":[{"id":"provider/real-model"}]}`), nil
	})
	config := modelVerificationTestConfig(t, "https://provider.example", "openai", modelHealthTestKey)
	handler := newModelHealthHandler(newModelHealthTestService(config, client, nil))
	failing := newModelHealthHandler(newModelHealthService("active-config.yaml", modelHealthOptions{
		loadConfig: func(string) (*routerconfig.RouterConfig, error) {
			return nil, errors.New("secret config path and value")
		},
	}))

	for _, test := range []struct {
		name    string
		handler http.HandlerFunc
		method  string
		target  string
		status  int
		code    string
	}{
		{name: "one model", handler: handler, method: http.MethodGet, target: modelHealthPath + "?model=logical-model&refresh=1", status: http.StatusOK},
		{name: "wrong method", handler: handler, method: http.MethodPost, target: modelHealthPath, status: http.StatusMethodNotAllowed, code: "method_not_allowed"},
		{name: "unknown model", handler: handler, method: http.MethodGet, target: modelHealthPath + "?model=missing", status: http.StatusNotFound, code: "model_not_found"},
		{name: "empty model", handler: handler, method: http.MethodGet, target: modelHealthPath + "?model=", status: http.StatusBadRequest, code: "invalid_request"},
		{name: "config unavailable", handler: failing, method: http.MethodGet, target: modelHealthPath, status: http.StatusServiceUnavailable, code: "config_unavailable"},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			response := httptest.NewRecorder()
			test.handler(response, httptest.NewRequest(test.method, test.target, nil))
			if response.Code != test.status {
				t.Fatalf("status = %d, want %d: %s", response.Code, test.status, response.Body.String())
			}
			if response.Header().Get("Cache-Control") != "no-store" {
				t.Fatalf("Cache-Control = %q", response.Header().Get("Cache-Control"))
			}
			if test.status == http.StatusMethodNotAllowed && response.Header().Get("Allow") != http.MethodGet {
				t.Fatalf("Allow = %q", response.Header().Get("Allow"))
			}
			if strings.Contains(response.Body.String(), "secret config") {
				t.Fatalf("response leaks the config error: %s", response.Body.String())
			}
			if test.code != "" {
				var body ModelHealthErrorResponse
				if err := json.Unmarshal(response.Body.Bytes(), &body); err != nil || body.Error != test.code {
					t.Fatalf("error body = %s, want code %q", response.Body.String(), test.code)
				}
				return
			}
			var body ModelHealthResponse
			if err := json.Unmarshal(response.Body.Bytes(), &body); err != nil {
				t.Fatalf("decode response: %v", err)
			}
			if len(body.Models) != 1 || body.Models[0].Status != ModelHealthStatusHealthy {
				t.Fatalf("models = %+v", body.Models)
			}
		})
	}
}

func TestPercentile95(t *testing.T) {
	t.Parallel()

	twenty := make([]int64, 0, 20)
	for value := int64(20); value >= 1; value-- {
		twenty = append(twenty, value)
	}
	for _, test := range []struct {
		samples []int64
		want    int64
	}{
		{samples: []int64{42}, want: 42},
		{samples: []int64{100, 1}, want: 100},
		{samples: twenty, want: 19},
	} {
		if got := percentile95(test.samples); got != test.want {
			t.Fatalf("percentile95(%v) = %d, want %d", test.samples, got, test.want)
		}
	}
}
