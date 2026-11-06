package classification

import (
	"sync"
	"testing"

	"github.com/prometheus/client_golang/prometheus/testutil"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
)

func matchCount(signalType, signalName string) float64 {
	return testutil.ToFloat64(metrics.SignalMatchTotal.WithLabelValues(signalType, signalName))
}

func newMetricScopeResults() *SignalResults {
	return &SignalResults{
		SignalConfidences: map[string]float64{},
		SignalValues:      map[string]float64{},
		SignalErrors:      map[string]string{},
		Metrics:           &SignalMetricsCollection{},
	}
}

// A recipe-local rule name is only unique inside its recipe, so every signal
// metric has to carry the scoped name. Two recipes that both declare
// "tenant_tier" would otherwise share one counter.
func TestMetadataSignalMatchMetricUsesRecipeScope(t *testing.T) {
	const ruleName = "metric_scope_metadata_rule"
	equals := "gold"
	classifier := &Classifier{Config: &config.RouterConfig{
		RoutingScope: "metric-scope-recipe",
		IntelligentRouting: config.IntelligentRouting{
			Signals: config.Signals{MetadataRules: []config.MetadataRule{{
				Name:      ruleName,
				Key:       "tier",
				Predicate: config.MetadataPredicate{Equals: &equals},
			}}},
		},
	}}
	scoped := config.RoutingNamespaceKey(classifier.Config.RoutingScope, ruleName)
	if scoped == ruleName {
		t.Fatalf("test recipe produced no scope for %q", ruleName)
	}
	beforeScoped := matchCount(config.SignalTypeMetadata, scoped)
	beforeLocal := matchCount(config.SignalTypeMetadata, ruleName)

	var mu sync.Mutex
	classifier.evaluateMetadataSignal(
		newMetricScopeResults(),
		&mu,
		RequestFacts{Metadata: map[string]string{"tier": "gold"}},
		map[string]bool{config.SignalTypeMetadata + ":" + ruleName: true},
	)

	if got := matchCount(config.SignalTypeMetadata, scoped) - beforeScoped; got != 1 {
		t.Fatalf("scoped match counter moved by %v, want 1", got)
	}
	if got := matchCount(config.SignalTypeMetadata, ruleName) - beforeLocal; got != 0 {
		t.Fatalf("unscoped match counter moved by %v, want 0", got)
	}
}
