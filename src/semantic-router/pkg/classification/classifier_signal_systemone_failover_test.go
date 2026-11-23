package classification

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"slices"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/serving"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice/runtimetest"
)

const decisionFailoverRouteConfig = `version: v0.5
providers:
  defaults: {model: general}
  models:
    - name: general
      backend_refs: [{name: general, endpoint: "127.0.0.1:8000", protocol: http}]
    - name: tool-model
      backend_refs: [{name: tools, endpoint: "127.0.0.1:8001", protocol: http}]
recipes:
  - name: default
    signals:
      systemone:
        - name: needs_tools
          timeout_ms: 2000
          question: {type: noul, instructions: "Does this request need tools?"}
    decisions:
      - name: tools
        priority: 100
        rules:
          operator: AND
          conditions: [{type: systemone, name: needs_tools, on_error: no_match}]
        modelRefs: [{model: tool-model}]
      - name: ordinary
        modelRefs: [{model: general}]
global:
  model_catalog:
    deployments:
      primary:
        provider: systemone
        schema: vllm-sr/5.0
        endpoint: %s/v1/systemone
        model: primary-model
        capabilities: {question_types: [noul]}
        auth: {api_key_env: TEST_FAILOVER_PRIMARY_KEY}
        timeout_ms: 250
        fallbacks: [spare, cpu-backup]
      spare:
        provider: systemone
        endpoint: %s/v1/systemone
        model: spare-model
        auth: {api_key_env: TEST_FAILOVER_SPARE_KEY}
      cpu-backup:
        provider: model_runtime
        endpoint: %s
        model: backup-model
        timeout_ms: 600
        auth: {api_key_env: TEST_FAILOVER_CPU_KEY}
`

// Use both real client adapters and the routing engine: only the last
// deployment says yes, so the request goes to tool-model only after failover.
func TestSystemOneSignalFailoverRoutesThroughAttachedRuntime(t *testing.T) {
	t.Setenv("TEST_FAILOVER_PRIMARY_KEY", "primary-test-key")
	t.Setenv("TEST_FAILOVER_SPARE_KEY", "spare-test-key")
	t.Setenv("TEST_FAILOVER_CPU_KEY", "cpu-test-key")
	var status atomic.Int32
	var backupStatus atomic.Int32
	var mu sync.Mutex
	var order []string
	seen := func(name string) {
		mu.Lock()
		defer mu.Unlock()
		order = append(order, name)
	}
	primary := newFailoverTestServer(t, "primary-test-key", func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodPost {
			seen("primary")
		}
		writeFailoverTestAnswer(w, r, int(status.Load()))
	})
	spare := newFailoverTestServer(t, "spare-test-key", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.NotFound(w, r)
			return
		}
		seen("spare")
		http.Error(w, "unavailable", http.StatusServiceUnavailable)
	})
	native := runtimetest.New(runtimetest.Model{ID: "backup-model"})
	handler := native.Handler()
	cpu := newFailoverTestServer(t, "cpu-test-key", func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/v1/decisions" || r.URL.Path == "/v1/bundle" {
			seen("cpu-backup")
			if code := int(backupStatus.Load()); code != 0 {
				http.Error(w, "backup did not answer", code)
				return
			}
		}
		handler.ServeHTTP(w, r)
	})
	classifier := newFailoverTestClassifier(t, primary.URL, spare.URL, cpu.URL)
	for _, test := range []struct {
		name, route, signalError string
		status, backupStatus     int32
		failover                 bool
	}{
		{name: "healthy", status: http.StatusOK, route: "ordinary"},
		{name: "unavailable", status: http.StatusServiceUnavailable, route: "tools", failover: true},
		{name: "upstream failure", status: http.StatusBadGateway, route: "tools", failover: true},
		{name: "overloaded", status: http.StatusTooManyRequests, route: "tools", failover: true},
		{name: "attempt timed out", status: 0, route: "tools", failover: true},
		{name: "both backups unavailable", status: http.StatusServiceUnavailable, backupStatus: http.StatusServiceUnavailable, route: "ordinary", failover: true, signalError: "systemone_unavailable"},
		{name: "authorization refused", status: http.StatusUnauthorized, route: "ordinary", signalError: "systemone_rejected"},
		{name: "malformed answer", status: -1, route: "ordinary", signalError: "systemone_failed"},
	} {
		t.Run(test.name, func(t *testing.T) {
			mu.Lock()
			order = nil
			mu.Unlock()
			status.Store(test.status)
			backupStatus.Store(test.backupStatus)
			ctx, cancel := context.WithTimeout(t.Context(), 5*time.Second)
			defer cancel()
			state := "Read a document: " + test.name
			signals, err := classifier.EvaluateAllSignalsWithHeaders(SignalEvaluationInput{Text: state, RequestFacts: RequestFacts{Context: ctx}})
			if err != nil {
				t.Fatal(err)
			}
			decision, err := classifier.EvaluateDecisionWithEngine(signals)
			if err != nil || decision == nil || decision.Decision.Name != test.route {
				t.Fatalf("decision = %+v, err = %v, signals = %v; want %s", decision, err, signals.SignalErrors, test.route)
			}
			if got := signals.SignalErrors["systemone:needs_tools"]; got != test.signalError {
				t.Fatalf("signal error = %q, want %q", got, test.signalError)
			}
			mu.Lock()
			asks := slices.Clone(order)
			mu.Unlock()
			want := []string{"primary"}
			if test.failover {
				want = append(want, "spare", "cpu-backup")
			}
			if !slices.Equal(asks, want) {
				t.Fatalf("attempts = %v, want %v", asks, want)
			}
			if test.failover && test.backupStatus == 0 {
				calls := native.Decisions()
				last := calls[len(calls)-1]
				if last.Model == nil || *last.Model != "backup-model" || last.State != state || len(last.Questions) != 1 {
					t.Fatalf("backup received the wrong model, state, or questions: %+v", last)
				}
				if last.Options == nil || last.Options.DeadlineMs == nil || *last.Options.DeadlineMs <= 0 || *last.Options.DeadlineMs > 600 {
					t.Fatalf("backup did not receive its own timeout within the chain's deadline: %+v", last.Options)
				}
			}
		})
	}
}

func newFailoverTestServer(t *testing.T, key string, handler http.HandlerFunc) *httptest.Server {
	t.Helper()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "Bearer "+key {
			http.Error(w, "wrong test credential", http.StatusUnauthorized)
			return
		}
		handler(w, r)
	}))
	t.Cleanup(server.Close)
	return server
}

func writeFailoverTestAnswer(w http.ResponseWriter, r *http.Request, status int) {
	if r.Method != http.MethodPost {
		http.NotFound(w, r)
		return
	}
	switch status {
	case 0:
		_, _ = io.Copy(io.Discard, r.Body)
		<-r.Context().Done()
	case -1:
		_, _ = io.WriteString(w, "not a decision response")
	case http.StatusOK:
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"model":"primary-model","answers":{"systemone:needs_tools":{"type":"noul","noul":0.1}}}`)
	default:
		http.Error(w, "primary did not answer", status)
	}
}

func newFailoverTestClassifier(t *testing.T, primary, spare, cpu string) *Classifier {
	t.Helper()
	cfg, err := config.ParseYAMLBytes([]byte(fmt.Sprintf(decisionFailoverRouteConfig, primary, spare, cpu)))
	if err != nil {
		t.Fatal(err)
	}
	lease, err := modelservice.NewManager().Acquire(cfg)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = lease.Close() })
	ctx, cancel := context.WithTimeout(t.Context(), 5*time.Second)
	defer cancel()
	if _, err := lease.Card(ctx, "cpu-backup"); err != nil {
		t.Fatalf("backup not in the runtime plan or not ready: %v", err)
	}
	remote, err := modelservice.NewRemoteSystemOne(cfg)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = remote.Close() })
	c, err := NewClassifier(cfg, RecipeRuntimeOptions{Runtime: serving.New(lease, nil), SystemOne: remote})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = c.Close() })
	return c
}

type failoverTestDecider func(context.Context, string, modelservice.Request) (modelservice.Response, error)

func (d failoverTestDecider) Decide(ctx context.Context, deployment string, request modelservice.Request) (modelservice.Response, error) {
	return d(ctx, deployment, request)
}

func TestSystemOneSignalFailoverSharesCallerDeadline(t *testing.T) {
	for _, stopped := range []bool{false, true} {
		t.Run(fmt.Sprintf("caller cancelled: %t", stopped), func(t *testing.T) {
			cfg := &config.RouterConfig{}
			cfg.ModelDeployments = map[string]config.ModelDeployment{
				"primary": {Provider: config.ModelRuntimeProvider, TimeoutMs: 50},
				"backup":  {Provider: config.ModelRuntimeProvider, TimeoutMs: 60000},
			}
			c := &Classifier{Config: cfg}
			ctx, cancel := context.WithTimeout(t.Context(), 30*time.Second)
			defer cancel()
			deadline, _ := ctx.Deadline()
			var calls []string
			c.SetDecisionDecider(failoverTestDecider(func(attempt context.Context, deployment string, _ modelservice.Request) (modelservice.Response, error) {
				calls = append(calls, deployment)
				cutoff, bounded := attempt.Deadline()
				if deployment == "primary" {
					if !bounded || !cutoff.Before(deadline) {
						t.Fatalf("primary attempt deadline = %v, want less than %v", cutoff, deadline)
					}
					if stopped {
						cancel()
					}
					return modelservice.Response{}, context.DeadlineExceeded
				}
				if !bounded || !cutoff.Equal(deadline) {
					t.Fatalf("backup attempt deadline = %v, want original deadline %v", cutoff, deadline)
				}
				return modelservice.Response{Model: "backup"}, nil
			}))
			call := &systemOneCall{fallbackChain: []string{"primary", "backup"}, rules: []askedRule{systemOneAsked(config.SystemOneRule{})}}
			answer, err := c.sendSystemOneFailover(ctx, call, modelservice.Request{State: "x"})
			if stopped {
				if !errors.Is(err, context.Canceled) || !slices.Equal(calls, []string{"primary"}) {
					t.Fatalf("cancelled caller tried %v and returned %v", calls, err)
				}
			} else if err != nil || answer.Model != "backup" || !slices.Equal(calls, []string{"primary", "backup"}) {
				t.Fatalf("attempt timeout should try backup: tried %v, answer %+v, err %v", calls, answer, err)
			}
		})
	}
}
