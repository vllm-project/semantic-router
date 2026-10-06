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

func extractionCount(signalType, signalName string) float64 {
	return testutil.ToFloat64(metrics.SignalExtractionTotal.WithLabelValues(signalType, signalName))
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

func TestGenericClassifierSignalMetricsUseRecipeScope(t *testing.T) {
	const ruleName = "metric_scope_classifier_rule"
	classifier := &Classifier{
		Config: &config.RouterConfig{
			RoutingScope: "metric-scope-recipe",
			IntelligentRouting: config.IntelligentRouting{
				Signals: config.Signals{ClassifierRules: []config.ClassifierSignalRule{{
					Name:   ruleName,
					Type:   "llm",
					Model:  "risk-model",
					Labels: []string{"SAFE", "RISKY"},
				}}},
			},
		},
		genericClassifiers: map[string]labelClassifier{
			ruleName: fakeLabelClassifier{result: labelClassification{
				Scores: map[string]float64{"SAFE": 0, "RISKY": 1},
			}},
		},
	}
	scopedRule := config.RoutingNamespaceKey(classifier.Config.RoutingScope, ruleName)
	label := ruleName + ":RISKY"
	scopedLabel := config.RoutingNamespaceKey(classifier.Config.RoutingScope, label)
	beforeExtraction := extractionCount(config.SignalTypeClassifier, scopedRule)
	beforeMatch := matchCount(config.SignalTypeClassifier, scopedLabel)
	beforeLocalExtraction := extractionCount(config.SignalTypeClassifier, ruleName)
	beforeLocalMatch := matchCount(config.SignalTypeClassifier, label)

	var mu sync.Mutex
	classifier.evaluateGenericClassifierSignals(
		newMetricScopeResults(),
		&mu,
		"text",
		map[string]bool{config.SignalTypeClassifier + ":" + ruleName: true},
		t.Context(),
	)

	if got := extractionCount(config.SignalTypeClassifier, scopedRule) - beforeExtraction; got != 1 {
		t.Fatalf("scoped extraction counter moved by %v, want 1", got)
	}
	if got := matchCount(config.SignalTypeClassifier, scopedLabel) - beforeMatch; got != 1 {
		t.Fatalf("scoped match counter moved by %v, want 1", got)
	}
	if got := extractionCount(config.SignalTypeClassifier, ruleName) - beforeLocalExtraction; got != 0 {
		t.Fatalf("unscoped extraction counter moved by %v, want 0", got)
	}
	if got := matchCount(config.SignalTypeClassifier, label) - beforeLocalMatch; got != 0 {
		t.Fatalf("unscoped match counter moved by %v, want 0", got)
	}
}

// Response-stage evaluators live in extproc and reach the scoping through the
// exported wrapper, so the wrapper has to scope exactly like the in-package
// helper it forwards to.
func TestExportedRecordSignalExtractionUsesRecipeScope(t *testing.T) {
	const signalName = "metric_scope_exported_rule"
	classifier := &Classifier{Config: &config.RouterConfig{RoutingScope: "metric-scope-recipe"}}
	scoped := config.RoutingNamespaceKey(classifier.Config.RoutingScope, signalName)
	before := extractionCount(config.SignalTypeJailbreak, scoped)
	beforeLocal := extractionCount(config.SignalTypeJailbreak, signalName)

	classifier.RecordSignalExtraction(config.SignalTypeJailbreak, signalName, 0.01)

	if got := extractionCount(config.SignalTypeJailbreak, scoped) - before; got != 1 {
		t.Fatalf("scoped extraction counter moved by %v, want 1", got)
	}
	if got := extractionCount(config.SignalTypeJailbreak, signalName) - beforeLocal; got != 0 {
		t.Fatalf("unscoped extraction counter moved by %v, want 0", got)
	}
}
