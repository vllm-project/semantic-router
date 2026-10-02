package classification

import (
	"context"
	"errors"
	"reflect"
	"slices"
	"sync"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
)

type signalFailureEmbeddingProvider struct {
	*embedding.FuncProvider
	imageErr   error
	imageCalls int
}

func (p *signalFailureEmbeddingProvider) EmbedImage(context.Context, []byte, int) ([]float32, error) {
	p.imageCalls++
	return []float32{1, 0}, p.imageErr
}

func TestEmbeddingSignalErrorsPreserveModalityBoundaries(t *testing.T) {
	for _, tc := range []struct {
		name                    string
		text, image             string
		failText, failImage     bool
		wantErrors, wantMatches []string
	}{
		{name: "text failure", text: "query", image: "YQ==", failText: true, wantErrors: []string{"embedding:implicit", "embedding:text"}, wantMatches: []string{"image"}},
		{name: "image failure", text: "query", image: "YQ==", failImage: true, wantErrors: []string{"embedding:image"}, wantMatches: []string{"implicit", "text"}},
		{name: "both failures", text: "query", image: "YQ==", failText: true, failImage: true, wantErrors: []string{"embedding:implicit", "embedding:text", "embedding:image"}},
		{name: "image only", image: "YQ==", failText: true, wantMatches: []string{"image"}},
		{name: "text only", text: "query", failImage: true, wantMatches: []string{"implicit", "text"}},
		{name: "no applicable input", text: "  ", image: "  ", failText: true, failImage: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			textCalls := 0
			fp, err := embedding.NewFuncProvider("synthetic", 2, func(_ context.Context, text string) ([]float32, error) {
				if text == "query" {
					textCalls++
					if tc.failText {
						return nil, errors.New("synthetic text failure")
					}
				}
				return []float32{1, 0}, nil
			})
			if err != nil {
				t.Fatal(err)
			}
			p := &signalFailureEmbeddingProvider{FuncProvider: fp}
			if tc.failImage {
				p.imageErr = errors.New("synthetic image failure")
			}
			rules := []config.EmbeddingRule{
				{Name: "implicit", Candidates: []string{"anchor"}, SimilarityThreshold: .5, AggregationMethodConfiged: config.AggregationMethodMax},
				{Name: "text", QueryModality: config.QueryModalityText, Candidates: []string{"anchor"}, SimilarityThreshold: .5, AggregationMethodConfiged: config.AggregationMethodMax},
				{Name: "image", QueryModality: config.QueryModalityImage, Candidates: []string{"anchor"}, SimilarityThreshold: .5, AggregationMethodConfiged: config.AggregationMethodMax},
				{Name: "audio", QueryModality: config.QueryModalityAudio, Candidates: []string{"anchor"}, SimilarityThreshold: .5, AggregationMethodConfiged: config.AggregationMethodMax},
			}
			ec, err := NewEmbeddingClassifierWithProvider(rules, config.HNSWConfig{PreloadEmbeddings: true}, p)
			if err != nil {
				t.Fatal(err)
			}
			if err = ec.WarmupCandidateEmbeddings(); err != nil {
				t.Fatal(err)
			}
			c := &Classifier{keywordEmbeddingClassifier: ec}
			results := newSignalResultsForTest()
			var mu sync.Mutex
			c.evaluateEmbeddingSignal(context.Background(), results, &mu, embeddingSignalInput{Text: tc.text, Image: tc.image}, nil)
			if len(results.SignalErrors) != len(tc.wantErrors) {
				t.Fatalf("errors=%v, want keys %v", results.SignalErrors, tc.wantErrors)
			}
			for _, key := range tc.wantErrors {
				if results.SignalErrors[key] != "embedding_evaluation_failed" {
					t.Fatalf("errors=%v", results.SignalErrors)
				}
			}
			slices.Sort(results.MatchedEmbeddingRules)
			slices.Sort(tc.wantMatches)
			if !slices.Equal(results.MatchedEmbeddingRules, tc.wantMatches) {
				t.Fatalf("matches=%v, want %v", results.MatchedEmbeddingRules, tc.wantMatches)
			}
			for _, name := range tc.wantMatches {
				if _, ok := results.SignalValues["embedding:"+name]; !ok {
					t.Fatalf("missing healthy score for %s", name)
				}
			}
			if tc.text != "query" && textCalls != 0 {
				t.Fatalf("unexpected text inference: %d", textCalls)
			}
			if tc.image != "YQ==" && p.imageCalls != 0 {
				t.Fatalf("unexpected image inference: %d", p.imageCalls)
			}
			if tc.failText && tc.text == "query" {
				assertFailedSignalCannotSelectNegation(t, c, results, config.SignalTypeEmbedding, "implicit")
			}
		})
	}
}

func assertFailedSignalCannotSelectNegation(t *testing.T, c *Classifier, results *SignalResults, kind, name string) {
	t.Helper()
	c.Config = &config.RouterConfig{IntelligentRouting: config.IntelligentRouting{Decisions: []config.Decision{{Name: "negated", Rules: config.RuleCombination{Operator: "NOT", Conditions: []config.RuleCombination{{Type: kind, Name: name}}, OnUnknown: config.RuleOnUnknownFailRequest}}}}}
	if _, err := c.EvaluateDecisionWithEngine(results); err == nil {
		t.Fatal("failed signal under NOT must remain unknown and fail_request")
	}
}

func TestReaskSignalFailureAndNonApplicability(t *testing.T) {
	for _, tc := range []struct {
		name, current, failText       string
		prior                         []string
		lookback                      int
		wantError, wantMatch, noCalls bool
	}{
		{name: "current failure", current: "current", prior: []string{"prior"}, failText: "current", wantError: true},
		{name: "prior failure", current: "current", prior: []string{"prior"}, failText: "prior", wantError: true},
		{name: "no history", current: "current", noCalls: true},
		{name: "empty history", current: "current", prior: []string{" ", ""}, noCalls: true},
		{name: "no current", current: " ", prior: []string{"prior"}, noCalls: true},
		{name: "too few prior turns", current: "current", prior: []string{"prior", " "}, lookback: 2, noCalls: true},
		{name: "healthy repeat", current: "current", prior: []string{"prior"}, wantMatch: true},
		{name: "healthy no match", current: "current", prior: []string{"unrelated"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			calls := 0
			provider, err := embedding.NewFuncProvider("synthetic", 2, func(_ context.Context, text string) ([]float32, error) {
				calls++
				if text == tc.failText {
					return nil, errors.New("synthetic reask failure")
				}
				if text == "unrelated" {
					return []float32{0, 1}, nil
				}
				return []float32{1, 0}, nil
			})
			if err != nil {
				t.Fatal(err)
			}
			rc, err := NewReaskClassifierWithProvider([]config.ReaskRule{{Name: "repeat", Threshold: .8, LookbackTurns: tc.lookback}, {Name: "longer", Threshold: .8, LookbackTurns: 3}}, "synthetic", provider)
			if err != nil {
				t.Fatal(err)
			}
			c := &Classifier{reaskClassifier: rc}
			results := newSignalResultsForTest()
			var mu sync.Mutex
			c.evaluateReaskSignal(results, &mu, tc.current, tc.prior)
			if tc.noCalls && calls != 0 {
				t.Fatalf("non-applicable reask made %d embedding calls", calls)
			}
			if tc.wantError {
				if !reflect.DeepEqual(results.SignalErrors, map[string]string{"reask:repeat": "reask_evaluation_failed"}) {
					t.Fatalf("errors=%v", results.SignalErrors)
				}
				assertFailedReaskCannotBecomeLowProjection(t, c, results)
			} else if len(results.SignalErrors) != 0 {
				t.Fatalf("normal no-match must not be an error: %v", results.SignalErrors)
			}
			if slices.Contains(results.MatchedReaskRules, "repeat") != tc.wantMatch {
				t.Fatalf("matches=%v, want repeat=%v", results.MatchedReaskRules, tc.wantMatch)
			}
			if tc.wantMatch && results.SignalValues["reask:repeat"] != 1 {
				t.Fatalf("repeat values=%v", results.SignalValues)
			}
		})
	}
}

func assertFailedReaskCannotBecomeLowProjection(t *testing.T, c *Classifier, results *SignalResults) {
	t.Helper()
	c.Config = &config.RouterConfig{IntelligentRouting: config.IntelligentRouting{Projections: config.Projections{
		Scores:   []config.ProjectionScore{{Name: "recovery", Method: "weighted_sum", Inputs: []config.ProjectionScoreInput{{Type: config.SignalTypeReask, Name: "repeat", Weight: 1}}}},
		Mappings: []config.ProjectionMapping{{Name: "band", Source: "recovery", Method: "threshold_bands", Outputs: []config.ProjectionMappingOutput{{Name: "low", LT: float64Ptr(.5)}, {Name: "high", GTE: float64Ptr(.5)}}}},
	}, Decisions: []config.Decision{{Name: "low", Rules: config.RuleCombination{Type: config.SignalTypeProjection, Name: "low", OnUnknown: config.RuleOnUnknownFailRequest}}}}}
	c.applyProjections(results)
	if len(results.MatchedProjectionRules) != 0 || results.SignalErrors["projection:low"] == "" {
		t.Fatalf("failed recovery became a valid band: matches=%v errors=%v", results.MatchedProjectionRules, results.SignalErrors)
	}
	if _, err := c.EvaluateDecisionWithEngine(results); err == nil {
		t.Fatal("projected failure must reach decision fail_request")
	}
}
