package router

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"sync/atomic"
	"testing"

	"github.com/vllm-project/semantic-router/dashboard/backend/auth"
	"github.com/vllm-project/semantic-router/dashboard/backend/config"
	"github.com/vllm-project/semantic-router/dashboard/backend/handlers"
	"github.com/vllm-project/semantic-router/dashboard/backend/observability"
)

func TestTraceSummariesRouteSummarizesJaegerSearchesForLogsReaders(t *testing.T) {
	const traceID = "4bf92f3577b34da6a3ce929d0e0e4736"
	var searches atomic.Int32
	jaeger := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/traces" || r.URL.Query().Get("service") != "vllm-sr" {
			http.NotFound(w, r)
			return
		}
		searches.Add(1)
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"data":[{"traceID":"`+traceID+`","spans":[{"spanID":"a1","operationName":"semantic_router.request",`+
			`"references":[],"startTime":1000,"duration":2000,"processID":"p1","tags":[]}],"processes":{"p1":{"serviceName":"vllm-sr"}}}],"errors":null}`)
	}))
	defer jaeger.Close()

	store, err := auth.NewStore(filepath.Join(t.TempDir(), "auth.db"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = store.Close() })
	service := auth.NewService(store, "trace-summaries-route-secret", 1)
	session := func(email, role string) string {
		t.Helper()
		const password = "trace-summaries-route-test"
		hash, err := service.HashPassword(password)
		if err != nil {
			t.Fatal(err)
		}
		if _, err = store.CreateUser(t.Context(), email, email, hash, role, "active"); err != nil {
			t.Fatal(err)
		}
		token, _, err := service.Login(t.Context(), email, password)
		if err != nil {
			t.Fatal(err)
		}
		return token
	}
	serve := func(jaegerURL string) (http.Handler, *auth.PolicyMux) {
		mux := auth.NewPolicyMux()
		registerJaegerRoutes(mux, &config.Config{JaegerURL: jaegerURL})
		mux.Seal()
		return wrapWithAuth(mux, service, mux), mux
	}
	get := func(handler http.Handler, token string) *httptest.ResponseRecorder {
		t.Helper()
		request := httptest.NewRequest(http.MethodGet, observability.TraceSummariesPath+"?service=vllm-sr", nil)
		if token != "" {
			request.Header.Set("Authorization", "Bearer "+token)
		}
		response := httptest.NewRecorder()
		handler.ServeHTTP(response, request)
		return response
	}

	configured, policies := serve(jaeger.URL)
	policy, lookup := policies.LookupRoutePolicy(http.MethodGet, observability.TraceSummariesPath)
	if lookup != auth.RouteFound || policy.Permission != auth.PermLogsRead || policy.ResourceOwner != auth.ResourceOwnerObservability ||
		policy.Sensitivity != auth.SensitivitySensitive || policy.Operation == nil {
		t.Fatalf("policy = %+v, lookup = %v; want the Jaeger proxy's logs.read observability policy with an operation", policy, lookup)
	}

	operator := session("operator@example.test", auth.RoleWrite)
	response := get(configured, operator)
	var summaries handlers.TraceSummariesResponse
	if response.Code != http.StatusOK || json.Unmarshal(response.Body.Bytes(), &summaries) != nil ||
		len(summaries.Data) != 1 || summaries.Data[0].TraceID != traceID ||
		summaries.Data[0].RootOperation != "semantic_router.request" || summaries.Data[0].DurationMicros != 2000 {
		t.Fatalf("logs reader: status = %d, body = %s", response.Code, response.Body)
	}
	for token, want := range map[string]int{session("reader@example.test", auth.RoleRead): http.StatusForbidden, "": http.StatusUnauthorized} {
		if denied := get(configured, token); denied.Code != want {
			t.Fatalf("caller without logs.read: status = %d, want %d", denied.Code, want)
		}
	}
	if got := searches.Load(); got != 1 {
		t.Fatalf("Jaeger searches = %d, want 1", got)
	}

	unconfigured, _ := serve("")
	response = get(unconfigured, operator)
	var failure handlers.TraceSummariesErrorResponse
	if response.Code != http.StatusServiceUnavailable || json.Unmarshal(response.Body.Bytes(), &failure) != nil ||
		failure.Error != "jaeger_not_configured" {
		t.Fatalf("without Jaeger: status = %d, body = %s", response.Code, response.Body)
	}
}
