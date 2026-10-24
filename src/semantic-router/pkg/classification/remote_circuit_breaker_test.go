package classification

import (
	"bufio"
	"context"
	"fmt"
	"io"
	"net"
	"net/http"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus/testutil"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/connector"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/tasks"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
)

func TestRemoteCircuitBreakerSkipPreservesSignalPolicies(t *testing.T) {
	calls := 0
	remote, err := connector.New("http://classifier.invalid", func(_ context.Context, req *http.Request) error {
		calls++
		// A deterministic transport failure without a socket or remote service.
		req.URL.Scheme = "unavailable"
		return nil
	}, connector.Options{
		AttemptTimeout: time.Second, MaxRequestBytes: 1024, MaxResponseBytes: 1024, MaxErrorBytes: 64,
		CircuitBreaker: &connector.CircuitBreakerOptions{Name: t.Name(), ConsecutiveFailures: 1, OpenInterval: time.Minute},
	})
	require.NoError(t, err)
	defer remote.Close()
	backend := &HTTPClassifierInference{connector: remote, timeout: time.Second, mapping: newDeclaredLabelMapping([]string{"benign", "jailbreak"})}
	_, _ = backend.Classify(context.Background(), "trip")
	c := &Classifier{}
	rule := config.ClassifierSignalRule{Name: "risk", Labels: []string{"benign", "jailbreak"}}
	out := observationResults()
	out.SignalErrors = map[string]string{}
	c.evaluateGenericClassifierRule(context.Background(), out, &sync.Mutex{}, "skip", rule, &sequenceLabelClassifier{backend: backend, labels: rule.Labels})
	if out.SignalErrors["classifier:risk"] != genericClassifierTimeoutCode || len(out.SignalValues) != 0 || calls != 1 {
		t.Fatalf("skip: calls=%d errors=%v values=%v", calls, out.SignalErrors, out.SignalValues)
	}
	for _, policy := range []string{config.OnErrorBlock, config.OnErrorAllow} {
		guard := observationClassifier([]config.JailbreakRule{{Name: "guard", Threshold: .5}}, backend, policy)
		guard.Config.PromptGuard.Backend = &config.RemoteClassifierBackend{Protocol: config.RemoteClassifierProtocolHTTPClassify, Model: "remote"}
		result := observationResults()
		guard.evaluateJailbreakSignalPieces(context.Background(), result, &sync.Mutex{}, []string{"skip"}, nil)
		if result.JailbreakDetected != (policy == config.OnErrorBlock) || result.JailbreakScoreAvailable || result.SignalErrors["jailbreak:guard"] != jailbreakEvaluationFailedCode {
			t.Fatalf("on_error %s: %+v", policy, result)
		}
	}
	require.Equal(t, 1, calls, "open breaker issued RPCs")
}

func TestCircuitBreakerLeavesLocalClassifiersUntouched(t *testing.T) {
	cfg := &config.RouterConfig{ExternalModels: []config.ExternalModelConfig{{CircuitBreaker: &config.ClassifierCircuitBreaker{Enabled: true}}}}
	_, guard, err := buildJailbreakDependencies(cfg, nil)
	require.NoError(t, err)
	require.IsType(t, &ownedSequenceBackend{}, guard)
	_, pii, err := buildPIIDependencies(cfg, nil)
	require.NoError(t, err)
	require.IsType(t, &ownedTokenBackend{}, pii)
}

func TestRemoteCircuitBreakerPreservesOverallDeadline(t *testing.T) {
	for _, family := range []string{"labels", "score", "tokens"} {
		for _, enabled := range []bool{false, true} {
			for _, callerDeadline := range []bool{false, true} {
				t.Run(fmt.Sprintf("%s/enabled=%t/caller=%t", family, enabled, callerDeadline), func(t *testing.T) {
					const timeout = 20 * time.Millisecond
					var calls atomic.Int32
					remoteClassifierPipeTransport(t, func(*http.Request) *http.Response {
						calls.Add(1)
						return nil // Stall before headers until the transport closes the connection.
					})
					threshold := 1
					cfg := &config.ExternalModelConfig{
						ModelEndpoint:  config.ClassifierVLLMEndpoint{Address: "classifier.invalid", Port: 80},
						CircuitBreaker: &config.ClassifierCircuitBreaker{Enabled: enabled, ConsecutiveFailures: &threshold},
					}
					var call func(context.Context) error
					switch family {
					case "labels":
						backend, err := newHTTPClassifierInference(cfg, newDeclaredLabelMapping([]string{"safe", "unsafe"}), timeout)
						require.NoError(t, err)
						defer backend.Close()
						call = func(ctx context.Context) error { _, err := backend.Classify(ctx, "hung"); return err }
					case "score":
						backend, err := newScoringHTTPBackend(cfg, timeout)
						require.NoError(t, err)
						defer backend.(*scoringHTTPBackend).Close()
						call = func(ctx context.Context) error { _, err := backend.Score(ctx, "hung"); return err }
					case "tokens":
						backend, err := newHTTPTokenClassifierInference(cfg, tasks.TokenLabelSet{Labels: []string{"O", "PERSON"}}, timeout)
						require.NoError(t, err)
						defer backend.Close()
						call = func(ctx context.Context) error { _, err := backend.ClassifyTokens(ctx, "hung"); return err }
					}
					for i := 0; i < 2; i++ {
						ctx, cancel := context.Background(), func() {}
						if callerDeadline {
							ctx, cancel = context.WithTimeout(context.Background(), 5*time.Millisecond)
						}
						start := time.Now()
						err := call(ctx)
						elapsed := time.Since(start)
						cancel()
						require.ErrorIs(t, err, context.DeadlineExceeded)
						require.Less(t, elapsed, 60*time.Millisecond, "total deadline expanded by retries/backoff")
						want := i + 1
						if enabled && !callerDeadline {
							want = 1
						}
						require.Equal(t, int32(want), calls.Load(), "owned timeout must count, caller timeout stays neutral")
					}
				})
			}
		}
	}
}

func TestRemoteCircuitBreakerPreservesChatCatalogTimeout(t *testing.T) {
	for _, enabled := range []bool{false, true} {
		t.Run(fmt.Sprint(enabled), func(t *testing.T) {
			remoteClassifierPipeTransport(t, func(*http.Request) *http.Response { return nil })
			deadline := 2000
			cfg := &config.RouterConfig{ExternalModels: []config.ExternalModelConfig{{
				Name: t.Name(), ModelName: "guard", ModelRole: config.ModelRoleGuardrail,
				ModelEndpoint: config.ClassifierVLLMEndpoint{Address: "classifier.invalid", Port: 80}, TimeoutSeconds: 1,
				CircuitBreaker: &config.ClassifierCircuitBreaker{Enabled: enabled},
			}}}
			cfg.PromptGuard.Backend = &config.RemoteClassifierBackend{Model: t.Name(), Protocol: config.RemoteClassifierProtocolHTTPChat, DeadlineMs: &deadline}
			mapping := &JailbreakMapping{LabelToIdx: map[string]int{"safe": 0, "jailbreak": 1}, IdxToLabel: map[string]string{"0": "safe", "1": "jailbreak"}}
			backend, err := createJailbreakInference(&cfg.PromptGuard, cfg, mapping)
			require.NoError(t, err)
			defer backend.(io.Closer).Close()
			start := time.Now()
			_, err = backend.(LabelDecisionBackend).Decide(context.Background(), "hung")
			require.ErrorIs(t, err, context.DeadlineExceeded)
			elapsed := time.Since(start)
			require.GreaterOrEqual(t, elapsed, 800*time.Millisecond)
			require.Less(t, elapsed, 1500*time.Millisecond, "shorter catalog timeout must still bound chat transport")
		})
	}
}

// Use the real HTTP transport and adapters without binding a TCP listener.
func remoteClassifierPipeTransport(t *testing.T, respond func(*http.Request) *http.Response) {
	t.Helper()
	original := http.DefaultTransport
	transport := original.(*http.Transport).Clone()
	var pending sync.WaitGroup
	transport.DialContext = func(context.Context, string, string) (net.Conn, error) {
		client, peer := net.Pipe()
		pending.Add(1)
		go func() {
			defer pending.Done()
			defer peer.Close()
			req, err := http.ReadRequest(bufio.NewReader(peer))
			if err != nil {
				return
			}
			_, _ = io.Copy(io.Discard, req.Body)
			_ = req.Body.Close()
			response := respond(req)
			if response == nil {
				_, _ = io.Copy(io.Discard, peer)
				return
			}
			response.Close = true
			_ = response.Write(peer)
			_ = response.Body.Close()
		}()
		return client, nil
	}
	http.DefaultTransport = transport
	t.Cleanup(func() {
		http.DefaultTransport = original
		transport.CloseIdleConnections()
		pending.Wait()
	})
}

func TestRemoteCircuitBreakerDecodeFailureKeepsTransportRecovery(t *testing.T) {
	closed := metrics.RemoteConnectorBreakerTransitions.WithLabelValues(t.Name(), "closed")
	closedBefore := testutil.ToFloat64(closed)
	var calls atomic.Int32
	remoteClassifierPipeTransport(t, func(*http.Request) *http.Response {
		status, body := http.StatusOK, `[{"label":"safe","score":0.9},{"label":"unsafe","score":0.1}]`
		switch calls.Add(1) {
		case 1:
			status, body = http.StatusServiceUnavailable, ""
		case 2:
			body = "not JSON"
		}
		return &http.Response{StatusCode: status, Body: io.NopCloser(strings.NewReader(body))}
	})
	remote, err := connector.New("http://classifier.invalid", nil, connector.Options{
		AttemptTimeout: time.Second, MaxRequestBytes: 1024, MaxResponseBytes: 1024, MaxErrorBytes: 64,
		CircuitBreaker: &connector.CircuitBreakerOptions{Name: t.Name(), ConsecutiveFailures: 1, OpenInterval: time.Nanosecond},
	})
	require.NoError(t, err)
	defer remote.Close()
	backend := &HTTPClassifierInference{connector: remote, timeout: time.Second, mapping: newDeclaredLabelMapping([]string{"safe", "unsafe"})}
	_, err = backend.Classify(context.Background(), "trip")
	require.Error(t, err)
	_, err = backend.Classify(context.Background(), "probe")
	require.ErrorContains(t, err, "failed to parse http_classify response")
	require.Equal(t, closedBefore+1, testutil.ToFloat64(closed), "decode failure must not undo probe recovery")
	_, err = backend.Classify(context.Background(), "recovered")
	require.NoError(t, err)
	require.Equal(t, int32(3), calls.Load())
}
