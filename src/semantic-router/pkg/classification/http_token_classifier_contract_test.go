package classification

import (
	"context"
	"sort"
	"strings"
	"sync"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// newRemotePIIClassifier wires a token_spans.v1 test provider into a Classifier
// through the same adapter production uses, with one PII rule and the given
// on_error policy.
func newRemotePIIClassifier(t *testing.T, onError string, respond func(inputs string) any) *Classifier {
	t.Helper()
	_, extCfg := newTokenSpansServer(t, respond)
	backend, err := newPIIHTTPBackend(extCfg, testPIIMapping(), 0)
	if err != nil {
		t.Fatal(err)
	}
	return newPIIRuleClassifier(t, onError, backend)
}

// newPIIRuleClassifier builds a Classifier around any PIIInference with one
// catch-all PII rule at threshold 0.5.
func newPIIRuleClassifier(t *testing.T, onError string, inference PIIInference) *Classifier {
	t.Helper()
	cfg := &config.RouterConfig{}
	cfg.PIIModel.ModelID = "test-pii-model"
	cfg.PIIModel.Threshold = 0.5
	cfg.PIIModel.OnError = onError
	cfg.PIIMappingPath = "test-pii-mapping-path"
	cfg.PIIRules = []config.PIIRule{{Name: "no_pii", Threshold: 0.5}}
	c, err := newClassifierWithOptions(cfg, withPII(testPIIMapping(), nil, inference))
	if err != nil {
		t.Fatal(err)
	}
	return c
}

func runPIISignal(c *Classifier, text string) *SignalResults {
	results := &SignalResults{
		Metrics:           &SignalMetricsCollection{},
		SignalConfidences: make(map[string]float64),
		SignalValues:      make(map[string]float64),
		SignalErrors:      make(map[string]string),
	}
	var mu sync.Mutex
	c.evaluatePIISignal(context.Background(), results, &mu, text, nil)
	sort.Strings(results.PIIEntities)
	sort.Strings(results.MatchedPIIRules)
	return results
}

// conflictingAliasSpans are spans that carry both spellings of a field with
// different values: malformed provider output, not a valid entity.
var conflictingAliasSpans = []struct {
	name string
	span map[string]any
	want string
}{
	{"label vs entity_group", map[string]any{"label": "PERSON", "entity_group": "EMAIL_ADDRESS", "text": "José Alvarez", "score": 0.9, "start": 8, "end": 20}, "conflicting label"},
	{"text vs word", map[string]any{"label": "PERSON", "text": "José Alvarez", "word": "Jose Alvarez", "score": 0.9, "start": 8, "end": 20}, "conflicting text"},
}

const conflictingAliasText = "Contact José Alvarez today."

// A conflicting alias pair rejects the whole response.
func TestHTTPTokenClassifierRejectsConflictingAliases(t *testing.T) {
	for _, tc := range conflictingAliasSpans {
		t.Run(tc.name, func(t *testing.T) {
			_, cfg := newTokenSpansServer(t, func(string) any { return []map[string]any{tc.span} })
			backend, err := newHTTPTokenClassifierInference(cfg, testPIIMapping(), 0)
			if err != nil {
				t.Fatal(err)
			}
			entities, err := backend.ClassifyTokens(conflictingAliasText)
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("want %q error, got err=%v entities=%v", tc.want, err, entities)
			}
		})
	}
	t.Run("agreeing aliases are accepted", func(t *testing.T) {
		_, cfg := newTokenSpansServer(t, func(string) any {
			return []map[string]any{{"label": "B-PERSON", "entity_group": "PERSON", "text": "José Alvarez", "word": "José Alvarez", "score": 0.9, "start": 8, "end": 20}}
		})
		backend, _ := newHTTPTokenClassifierInference(cfg, testPIIMapping(), 0)
		entities, err := backend.ClassifyTokens(conflictingAliasText)
		if err != nil || len(entities) != 1 || entities[0].EntityType != "PERSON" {
			t.Fatalf("agreeing aliases rejected: %v (%v)", err, entities)
		}
	})
}

// The rejection routes through on_error like any other backend failure: allow
// reads as a non-match, block fails closed, and the malformed label never
// reaches PIIEntities either way.
func TestConflictingAliasesRouteThroughOnError(t *testing.T) {
	for _, tc := range conflictingAliasSpans {
		for _, policy := range []struct {
			onError      string
			wantDetected bool
		}{{config.OnErrorAllow, false}, {config.OnErrorBlock, true}} {
			t.Run(tc.name+" under on_error "+policy.onError, func(t *testing.T) {
				c := newRemotePIIClassifier(t, policy.onError, func(string) any { return []map[string]any{tc.span} })
				results := runPIISignal(c, conflictingAliasText)
				if results.PIIDetected != policy.wantDetected {
					t.Fatalf("PIIDetected = %v, want %v (entities=%v)", results.PIIDetected, policy.wantDetected, results.PIIEntities)
				}
				for _, e := range results.PIIEntities {
					if e == "PERSON" || e == "EMAIL_ADDRESS" {
						t.Fatalf("a malformed span reached PIIEntities: %v", results.PIIEntities)
					}
				}
			})
		}
	}
}

// The envelope's model member is optional; when present it must name the
// configured model. The test provider is configured as llm_model_name
// "pii-spans".
func TestHTTPTokenClassifierModelIdentity(t *testing.T) {
	const text = "John Smith called yesterday."
	span := map[string]any{"label": "PERSON", "text": "John Smith", "score": 0.98, "start": 0, "end": 10}

	cases := []struct {
		name    string
		body    any
		wantErr string
	}{
		{"matching model", map[string]any{"spans": []any{span}, "model": "pii-spans"}, ""},
		{"model omitted", map[string]any{"spans": []any{span}}, ""},
		{"bare list carries no model", []any{span}, ""},
		{"different model", map[string]any{"spans": []any{span}, "model": "some-other-model"}, "names a different model"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, cfg := newTokenSpansServer(t, func(string) any { return tc.body })
			backend, err := newHTTPTokenClassifierInference(cfg, testPIIMapping(), 0)
			if err != nil {
				t.Fatal(err)
			}
			entities, err := backend.ClassifyTokens(text)
			if tc.wantErr == "" {
				if err != nil || len(entities) != 1 {
					t.Fatalf("want one entity, got err=%v entities=%v", err, entities)
				}
				return
			}
			if err == nil || !strings.Contains(err.Error(), tc.wantErr) {
				t.Fatalf("want %q error, got err=%v entities=%v", tc.wantErr, err, entities)
			}
		})
	}

	t.Run("wrong model fails closed under on_error block", func(t *testing.T) {
		c := newRemotePIIClassifier(t, config.OnErrorBlock, func(string) any {
			return map[string]any{"spans": []any{span}, "model": "some-other-model"}
		})
		results := runPIISignal(c, text)
		if !results.PIIDetected || len(results.PIIEntities) != 1 || results.PIIEntities[0] != PIIClassificationErrorType {
			t.Fatalf("want classification_error only, got detected=%v entities=%v", results.PIIDetected, results.PIIEntities)
		}
	})
}
