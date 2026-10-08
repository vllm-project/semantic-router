package modelservice

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus/testutil"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice/runtimetest"
)

func systemOneManager(t *testing.T, invoke http.HandlerFunc) *Manager {
	t.Helper()
	fake := runtimetest.New(runtimetest.Model{ID: "served"})
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/v1/systemone" {
			invoke(w, r)
			return
		}
		fake.Handler().ServeHTTP(w, r)
	}))
	t.Cleanup(upstream.Close)
	manager := NewManager()
	t.Cleanup(func() { _ = manager.Shutdown(context.Background()) })
	cfg := &config.RouterConfig{}
	cfg.ModelDeployments = map[string]config.ModelDeployment{"test-systemone": {Provider: config.ModelRuntimeProvider, Endpoint: upstream.URL, ServedName: "served"}}
	cfg.DecisionRules = []config.DecisionSignalRule{{Name: "test", Deployment: "test-systemone", Question: config.DecisionQuestion{Type: "noul", Instructions: "test?"}}}
	if err := manager.Reconcile(cfg); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	if _, err := manager.Published().Card(ctx, "test-systemone"); err != nil {
		t.Fatal(err)
	}
	return manager
}

func TestSystemOneRetainsDeploymentAndPreservesNativeResponse(t *testing.T) {
	started, finish := make(chan struct{}), make(chan struct{})
	native := `{"model":"served","answers":{"intent":{"type":"choice","choice":"yes","confidence":0.9,"probabilities":{"yes":0.9,"no":0.1}}},"spans":{"pii":[{"start":0,"end":2,"label":"person","text":"张三","probability":0.9}]},"sets":{"needs":{"selected":["code"],"probabilities":{"code":0.9}}},"usage":{"input_tokens":12,"output_tokens":0},"meta":{"profile":"exact"}}`
	manager := systemOneManager(t, func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var request map[string]json.RawMessage
		if json.Unmarshal(body, &request) != nil || string(request["model"]) != `"served"` || string(request["questions"]) != `{"z":{"type":"noul","instructions":"z?"},"a":{"type":"noul","instructions":"a?"}}` {
			t.Errorf("request altered: %s", body)
		}
		close(started)
		<-finish
		w.Header().Set("Server-Timing", "forward;dur=1, total;dur=2")
		_, _ = io.WriteString(w, native)
	})
	before := testutil.ToFloat64(requestsTotal.WithLabelValues("test-systemone", "ok"))
	done := make(chan SystemOneResult, 1)
	go func() {
		result, err := manager.SystemOne(context.Background(), "test-systemone", json.RawMessage(`{"model":"unauthorized-other","state":"张三","questions":{"z":{"type":"noul","instructions":"z?"},"a":{"type":"noul","instructions":"a?"}},"options":{"return_meta":true}}`))
		if err != nil {
			t.Error(err)
		}
		done <- result
	}()
	<-started
	if err := manager.Reconcile(&config.RouterConfig{}); err != nil {
		t.Fatal(err)
	}
	manager.mu.Lock()
	groups := len(manager.groups)
	manager.mu.Unlock()
	if groups != 1 {
		t.Fatalf("in-flight deployment released on reconcile: %d", groups)
	}
	close(finish)
	result := <-done
	if result.Status != 200 || string(result.Body) != native || result.ServerTiming == "" {
		t.Fatalf("result=%+v", result)
	}
	manager.mu.Lock()
	groups = len(manager.groups)
	manager.mu.Unlock()
	if groups != 0 {
		t.Fatalf("retained deployment leaked: %d", groups)
	}
	if got := testutil.ToFloat64(requestsTotal.WithLabelValues("test-systemone", "ok")); got != before+1 {
		t.Fatalf("diagnostic not observed: %v -> %v", before, got)
	}
}

func TestSystemOneBoundsErrorsRedirectsAndCancellation(t *testing.T) {
	for _, tc := range []struct {
		name   string
		status int
		body   string
		want   error
	}{
		{"native error", 429, `{"error":{"code":"overloaded","message":"busy"}}`, ErrOverloaded},
		{"redirect", 307, `{}`, ErrFailed},
		{"malformed", 200, `<html>bad</html>`, ErrFailed},
		{"oversized", 200, `"` + strings.Repeat("x", SystemOneResponseLimit) + `"`, ErrFailed},
	} {
		t.Run(tc.name, func(t *testing.T) {
			manager := systemOneManager(t, func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Location", "http://untrusted.invalid/")
				w.WriteHeader(tc.status)
				_, _ = io.WriteString(w, tc.body)
			})
			result, err := manager.SystemOne(context.Background(), "test-systemone", json.RawMessage(`{"state":"text","questions":{}}`))
			if !errors.Is(err, tc.want) {
				t.Fatalf("err=%v", err)
			}
			if tc.status == 429 && (result.Status != 429 || string(result.Body) != tc.body) {
				t.Fatalf("native error changed: %+v", result)
			}
		})
	}
	manager := systemOneManager(t, func(w http.ResponseWriter, r *http.Request) { _, _ = io.Copy(io.Discard, r.Body); <-r.Context().Done() })
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Millisecond)
	defer cancel()
	if _, err := manager.SystemOne(ctx, "test-systemone", json.RawMessage(`{}`)); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("cancellation: %v", err)
	}
	if _, err := manager.SystemOne(context.Background(), "http://untrusted.invalid", json.RawMessage(`{}`)); !errors.Is(err, ErrUnknownDeployment) {
		t.Fatalf("URL target accepted: %v", err)
	}
}
