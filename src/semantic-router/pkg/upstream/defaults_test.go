package upstream

import (
	"os"
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
	LBPolicy                         string  `yaml:"lb_policy"`
	LeastRequestChoiceCount          int     `yaml:"least_request_choice_count"`
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
		ModelConfig:   map[string]config.ModelParams{"m": {Reliability: config.ProviderReliability{RetryCount: 1}}},
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
