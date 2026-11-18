package upstream

import (
	"context"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"reflect"
	"sync/atomic"
	"syscall"
	"testing"
	"time"

	"gopkg.in/yaml.v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// reliabilityDefaults is testdata/reliability-defaults.yaml, the table the
// CLI's Envoy render test checks too.
type reliabilityDefaults struct {
	ListenerTimeout                  string  `yaml:"listener_timeout"`
	ConnectTimeout                   string  `yaml:"connect_timeout"`
	IdleConnectionTimeout            string  `yaml:"idle_connection_timeout"`
	LBPolicy                         string  `yaml:"lb_policy"`
	LeastRequestChoiceCount          int     `yaml:"least_request_choice_count"`
	RetryCount                       int     `yaml:"retry_count"`
	RetryOn                          string  `yaml:"retry_on"`
	RetryBackOffBase                 string  `yaml:"retry_back_off_base"`
	RetryBackOffMax                  string  `yaml:"retry_back_off_max"`
	HostSelectionRetryMaxAttempts    int     `yaml:"host_selection_retry_max_attempts"`
	MaxConnections                   int     `yaml:"max_connections"`
	MaxPendingRequests               int     `yaml:"max_pending_requests"`
	MaxRequests                      int     `yaml:"max_requests"`
	MaxRequestsWithRetriesOrEjection int     `yaml:"max_requests_with_retries_or_ejection"`
	MaxRetries                       int     `yaml:"max_retries"`
	RetryBudgetPercent               float64 `yaml:"retry_budget_percent"`
	RetryBudgetMinConcurrency        int     `yaml:"retry_budget_min_concurrency"`
	OutlierInterval                  string  `yaml:"outlier_interval"`
	BaseEjectionTime                 string  `yaml:"base_ejection_time"`
	MaxEjectionTime                  string  `yaml:"max_ejection_time"`
	MaxEjectionPercent               int     `yaml:"max_ejection_percent"`
	HealthCheckInterval              string  `yaml:"health_check_interval"`
	HealthCheckTimeout               string  `yaml:"health_check_timeout"`
	HealthCheckNoTrafficInterval     string  `yaml:"health_check_no_traffic_interval"`
	UnhealthyThreshold               int     `yaml:"unhealthy_threshold"`
	HealthyThreshold                 int     `yaml:"healthy_threshold"`
	HealthyPanicPercent              int     `yaml:"healthy_panic_percent"`

	// RetryOnConditions are the retry_on names a configuration may use.
	RetryOnConditions []string `yaml:"retry_on_conditions"`
}

func loadReliabilityDefaults(t *testing.T) reliabilityDefaults {
	t.Helper()
	data, err := os.ReadFile("testdata/reliability-defaults.yaml")
	if err != nil {
		t.Fatal(err)
	}
	var table reliabilityDefaults
	if err := yaml.Unmarshal(data, &table); err != nil {
		t.Fatal(err)
	}
	return table
}

func mustDuration(t *testing.T, raw string) time.Duration {
	t.Helper()
	d, err := time.ParseDuration(raw)
	if err != nil {
		t.Fatalf("defaults table duration %q: %v", raw, err)
	}
	return d
}

func TestNativeDefaultsMatchTheSharedTable(t *testing.T) {
	table := loadReliabilityDefaults(t)
	topology, err := Compile(&config.RouterConfig{BackendModels: config.BackendModels{
		VLLMEndpoints: []config.VLLMEndpoint{{Name: "e", Address: "10.0.0.1", Port: 8000, Model: "m"}},
		ModelConfig:   map[string]config.ModelParams{"m": {Reliability: config.ProviderReliability{RetryCount: countOf(1)}}},
	}})
	if err != nil {
		t.Fatal(err)
	}
	listener := topology.Listeners[0].Timeouts
	retry := topology.Clusters[0].Policy.Retry
	waits := newBackOff(&RetryPolicy{}, globalRand{})
	breakers := Breakers{RetryBudget: &RetryBudget{}}.withDefaults()
	outlier := OutlierSpec{}.withDefaults()
	health := HealthCheckSpec{}.withDefaults()

	checks := []struct {
		name      string
		got, want any
	}{
		{"listener route timeout", listener.Total, mustDuration(t, table.ListenerTimeout)},
		{"listener idle timeout", listener.Idle, mustDuration(t, table.ListenerTimeout)},
		{"connect timeout", builtinPolicy().Timeouts.Connect, mustDuration(t, table.ConnectTimeout)},
		{"idle connection timeout", idleConnTimeout, mustDuration(t, table.IdleConnectionTimeout)},
		{"lb policy", string(topology.Clusters[0].LBPolicy), table.LBPolicy},
		{"least request choices", leastRequestChoices, table.LeastRequestChoiceCount},
		{"retry_on default", config.DefaultProviderRetryOn, table.RetryOn},
		{"retry_on compiled", retry.On, ParseRetryOn(table.RetryOn)},
		{"back-off base", waits.next, mustDuration(t, table.RetryBackOffBase)},
		{"back-off max", waits.max, mustDuration(t, table.RetryBackOffMax)},
		{"host re-picks", hostSelectionRetries, table.HostSelectionRetryMaxAttempts},
		{"max connections", breakers.MaxConnections, table.MaxConnections},
		{"max pending", breakers.MaxPendingRequests, table.MaxPendingRequests},
		{"max requests", breakers.MaxRequests, table.MaxRequests},
		{"raised max requests", raisedMaxRequests, table.MaxRequestsWithRetriesOrEjection},
		{"max retries", breakers.MaxRetries, table.MaxRetries},
		{"retry budget percent", breakers.RetryBudget.Percent, table.RetryBudgetPercent},
		{"retry budget floor", breakers.RetryBudget.MinConcurrency, table.RetryBudgetMinConcurrency},
		{"outlier interval", outlier.Interval, mustDuration(t, table.OutlierInterval)},
		{"base ejection time", outlier.BaseEjectionTime, mustDuration(t, table.BaseEjectionTime)},
		{"max ejection time", outlier.MaxEjectionTime, mustDuration(t, table.MaxEjectionTime)},
		{"max ejection percent", outlier.MaxEjectionPercent, table.MaxEjectionPercent},
		{"health check interval", health.Interval, mustDuration(t, table.HealthCheckInterval)},
		{"health check timeout", health.Timeout, mustDuration(t, table.HealthCheckTimeout)},
		{"no-traffic interval", health.NoTrafficInterval, mustDuration(t, table.HealthCheckNoTrafficInterval)},
		{"unhealthy threshold", health.UnhealthyThreshold, table.UnhealthyThreshold},
		{"healthy threshold", health.HealthyThreshold, table.HealthyThreshold},
		{"panic threshold", healthyPanicPercent, table.HealthyPanicPercent},
	}
	for _, check := range checks {
		if check.got != check.want {
			t.Errorf("%s = %v, the shared table says %v", check.name, check.got, check.want)
		}
	}
}

const bareModelYAML = `
version: v0.5
providers:
  models:
    - name: m
      provider_model_id: m
      api_format: openai
%s      backend_refs:
        - {name: e, provider: vllm, endpoint: '%s'}
recipes:
  - name: default
`

// A reliability block that names no retries gets the table's: retries of an
// attempt the backend never received. An explicit retry_count: 0 turns them
// off, and retry_on alone still retries nothing.
func TestDefaultRetriesNeedNeitherRetryCountNorRetryOn(t *testing.T) {
	table := loadReliabilityDefaults(t)
	want := &RetryPolicy{NumRetries: table.RetryCount, On: ParseRetryOn(table.RetryOn)}
	// A blank retry_on is unset, as the CLI's Envoy render reads it.
	for _, reliability := range []string{"", "      reliability:\n        retry_on: '  '\n"} {
		topology := compileYAML(t, fmt.Sprintf(bareModelYAML, reliability, "10.0.0.1:8000"))
		bare := topology.cluster("m")
		if !reflect.DeepEqual(bare.Policy.Retry, want) || bare.Breakers.MaxRequests != table.MaxRequestsWithRetriesOrEjection {
			t.Fatalf("%q: retry %+v, max requests %d; want %+v and %d", reliability,
				bare.Policy.Retry, bare.Breakers.MaxRequests, want, table.MaxRequestsWithRetriesOrEjection)
		}
	}
	for _, block := range []string{"retry_count: 0", "retry_on: reset"} {
		reliability := "      reliability:\n        " + block + "\n"
		topology := compileYAML(t, fmt.Sprintf(bareModelYAML, reliability, "10.0.0.1:8000"))
		spec := topology.cluster("m")
		if spec.Policy.Retry != nil || spec.Breakers.MaxRequests != 0 {
			t.Fatalf("%s: retry %+v, max requests %d; want no retries", block, spec.Policy.Retry, spec.Breakers.MaxRequests)
		}
	}
}

// bareModelSet serves model m on server with the reliability a config gets
// when it sets none.
func bareModelSet(t *testing.T, server *httptest.Server, opts Options) *Set {
	t.Helper()
	topology := compileYAML(t, fmt.Sprintf(bareModelYAML, "", server.Listener.Addr().String()))
	set, err := New(topology, opts)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	t.Cleanup(func() { _ = set.Close(context.Background()) })
	return set
}

func TestDefaultRetryRecoversARefusedConnection(t *testing.T) {
	var received atomic.Int32
	server := backend(t, func(w http.ResponseWriter, r *http.Request) {
		received.Add(1)
		w.WriteHeader(http.StatusOK)
	})
	var dials atomic.Int32
	dial := func(ctx context.Context, network, address string) (net.Conn, error) {
		if dials.Add(1) == 1 {
			return nil, &net.OpError{Op: "dial", Net: network, Err: syscall.ECONNREFUSED}
		}
		return (&net.Dialer{}).DialContext(ctx, network, address)
	}
	resp, err := bareModelSet(t, server, Options{Dial: dial}).Do(t.Context(), post("m"))
	if err != nil {
		t.Fatalf("Do: %v", err)
	}
	_, _ = readAll(t, resp)
	if resp.StatusCode != http.StatusOK || len(resp.Attempts) != 2 || resp.Attempts[0].Err.Kind != KindConnectFailure {
		t.Fatalf("status %d after attempts %+v, want 200 after retrying the refused connection", resp.StatusCode, resp.Attempts)
	}
	if received.Load() != 1 {
		t.Fatalf("backend received %d requests, want 1", received.Load())
	}
}

func TestDefaultRetryNeverResendsAReceivedRequest(t *testing.T) {
	var received atomic.Int32
	server := backend(t, func(w http.ResponseWriter, r *http.Request) {
		received.Add(1)
		_, _ = io.Copy(io.Discard, r.Body)
		hijackAndClose(w)
	})
	resp, err := bareModelSet(t, server, Options{}).Do(t.Context(), post("m"))
	if err != nil {
		t.Fatalf("Do: %v", err)
	}
	assertLocalReply(t, resp, http.StatusServiceUnavailable,
		"upstream connect error or disconnect/reset before headers. reset reason: connection termination")
	if len(resp.Attempts) != 1 || received.Load() != 1 {
		t.Fatalf("%d attempts, backend received %d requests; want the one request, never resent",
			len(resp.Attempts), received.Load())
	}
}
