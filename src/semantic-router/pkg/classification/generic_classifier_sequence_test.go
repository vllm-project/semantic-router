package classification

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func toxicityRule() config.ClassifierSignalRule {
	return config.ClassifierSignalRule{
		Name:   "toxicity",
		Type:   config.ClassifierSignalTypeSequenceClassifier,
		Model:  "toxicity-endpoint",
		Labels: []string{"benign", "toxic"},
	}
}

func newTestSequenceClassifier(t *testing.T, server *httptest.Server, rule config.ClassifierSignalRule) labelClassifier {
	t.Helper()
	classifier, err := newSequenceLabelClassifier(rule, &config.ExternalModelConfig{
		ModelEndpoint: config.ClassifierVLLMEndpoint{Address: "placeholder", Port: 1},
		ModelRole:     config.ModelRoleClassification,
	})
	if err != nil {
		t.Fatalf("failed to construct classifier: %v", err)
	}
	classifier.(*sequenceLabelClassifier).backend.baseURL = server.URL
	return classifier
}

func TestDeclaredLabelMappingUsesConfigOrder(t *testing.T) {
	mapping := newDeclaredLabelMapping([]string{"benign", "toxic"})

	if got := mapping.LabelCount(); got != 2 {
		t.Errorf("LabelCount() = %d, want 2", got)
	}
	if index, ok := mapping.IndexForLabel("toxic"); !ok || index != 1 {
		t.Errorf("IndexForLabel(toxic) = %d, %t; want 1, true", index, ok)
	}
	if _, ok := mapping.IndexForLabel("absent"); ok {
		t.Error("IndexForLabel(absent) reported a match for an undeclared label")
	}
	if label, ok := mapping.LabelFromIndex(0); !ok || label != "benign" {
		t.Errorf("LabelFromIndex(0) = %q, %t; want benign, true", label, ok)
	}
	if _, ok := mapping.LabelFromIndex(2); ok {
		t.Error("LabelFromIndex(2) reported a match outside the declared labels")
	}
}

func TestSequenceLabelClassifierReportsTheModelDistribution(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_ = json.NewEncoder(w).Encode([]httpClassifyLabelScore{
			{Label: "toxic", Score: 0.73},
			{Label: "benign", Score: 0.27},
		})
	}))
	defer server.Close()

	classifier := newTestSequenceClassifier(t, server, toxicityRule())
	result, err := classifier.Classify(context.Background(), "some text")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if got := result.Scores["toxic"]; got < 0.72 || got > 0.74 {
		t.Errorf("toxic score = %v, want ~0.73", got)
	}
	if got := result.Scores["benign"]; got < 0.26 || got > 0.28 {
		t.Errorf("benign score = %v, want ~0.27", got)
	}
	if result.Rationale != "" {
		t.Errorf("rationale = %q, want empty for a sequence classifier", result.Rationale)
	}
}

func TestSequenceLabelClassifierRejectsIncompleteDistribution(t *testing.T) {
	// A response omitting a declared label would otherwise default that label
	// to 0.0 and silently under-report the signal.
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_ = json.NewEncoder(w).Encode([]httpClassifyLabelScore{
			{Label: "toxic", Score: 1.0},
		})
	}))
	defer server.Close()

	classifier := newTestSequenceClassifier(t, server, toxicityRule())
	if _, err := classifier.Classify(context.Background(), "some text"); err == nil {
		t.Fatal("expected an error when the response omits a declared label, got nil")
	}
}

func TestNewSequenceLabelClassifierRequiresTheExternalModel(t *testing.T) {
	if _, err := newSequenceLabelClassifier(toxicityRule(), nil); err == nil {
		t.Fatal("expected an error for an unconfigured external model, got nil")
	}
}

func TestBuildGenericClassifiersRejectsUnknownType(t *testing.T) {
	// Only reachable for a rule that bypassed config validation.
	builder := &classifierOptionBuilder{cfg: &config.RouterConfig{
		IntelligentRouting: config.IntelligentRouting{
			Signals: config.Signals{ClassifierRules: []config.ClassifierSignalRule{{
				Name:   "toxicity",
				Type:   "sequence-classifier",
				Labels: []string{"benign", "toxic"},
			}}},
		},
	}}

	if _, err := builder.buildGenericClassifiersOption(); err == nil {
		t.Fatal("expected an error for an unsupported classifier type, got nil")
	}
}
