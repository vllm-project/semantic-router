package classification

import (
	"context"
	"errors"
	"reflect"
	"strings"
	"sync"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
)

type complexityInputRecorder struct {
	mu     sync.Mutex
	texts  []string
	limits binding.Limits
}

func (p *complexityInputRecorder) Embed(_ context.Context, text string) ([]float32, error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.texts = append(p.texts, text)
	// This synthetic provider defines one whitespace-delimited token plus two
	// special tokens. Real native providers supply their own tokenizer counts.
	if err := p.limits.CheckInput(len(strings.Fields(text)) + 2); err != nil {
		return nil, err
	}
	if text == "easy prototype" {
		return []float32{0, 1}, nil
	}
	return []float32{1, 0}, nil
}

func (p *complexityInputRecorder) EmbedBatch(ctx context.Context, texts []string) ([][]float32, error) {
	result := make([][]float32, 0, len(texts))
	for _, text := range texts {
		vector, err := p.Embed(ctx, text)
		if err != nil {
			return nil, err
		}
		result = append(result, vector)
	}
	return result, nil
}

func (*complexityInputRecorder) Backend() string { return "candle" }
func (*complexityInputRecorder) Dimension() int  { return 2 }

func newComplexityInputClassifier(t *testing.T, fullContext bool, limit int) (*Classifier, *complexityInputRecorder) {
	t.Helper()
	cfg := &config.RouterConfig{}
	cfg.EmbeddingConfig.ModelType = "mmbert"
	cfg.EmbeddingConfig.FullContext = fullContext
	cfg.ComplexityRules = []config.ComplexityRule{{
		Name: "scope", Threshold: 0.1,
		Hard: config.ComplexityCandidates{Candidates: []string{"hard prototype"}},
		Easy: config.ComplexityCandidates{Candidates: []string{"easy prototype"}},
	}}
	provider := &complexityInputRecorder{limits: binding.Limits{
		ModelTokens: 32768, TaskTokens: 32768, DeploymentTokens: limit, Overflow: "reject",
	}}
	local, err := NewComplexityClassifier(cfg.ComplexityRules, "mmbert", config.PrototypeScoringConfig{}, provider)
	if err != nil {
		t.Fatal(err)
	}
	provider.texts = nil // Candidate initialization is complete before dispatch.
	return &Classifier{Config: cfg, complexityClassifier: local}, provider
}

func TestComplexityFullContextReachesEmbeddingProvider(t *testing.T) {
	full := strings.Repeat("neutral context detail ", 400)
	for _, tc := range []struct {
		name        string
		fullContext bool
		compressed  bool
		exempt      bool
	}{
		{name: "default retains representative sampling"},
		{name: "explicit full context preserves complete text", fullContext: true},
		{name: "compression exemption retains original", fullContext: true, compressed: true, exempt: true},
		{name: "compression remains independently configured", fullContext: true, compressed: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			classifier, provider := newComplexityInputClassifier(t, tc.fullContext, 8192)
			// A large explicit embedding budget alone must not opt into full text.
			classifier.Config.ModelBindings = map[string]config.ModelBinding{"embedding": {Deployment: "semantic"}}
			classifier.Config.ModelDeployments = map[string]config.ModelDeployment{"semantic": {
				Provider: "candle", Input: config.ModelInputBudget{MaxTokens: 8192, Overflow: "reject"},
			}}
			input, original := full, ""
			if tc.compressed {
				input, original = "compressed summary", full
			}
			results := classifier.EvaluateAllSignalsWithContext(input, full, input, nil, nil, false, true,
				original, map[string]bool{config.SignalTypeComplexity: tc.exempt}, ConversationFacts{}, "")
			want := full
			if tc.compressed && !tc.exempt {
				want = input
			}
			if !tc.fullContext {
				want = textForRoutingSignal(config.SignalTypeComplexity, want)
				if want == full || !strings.Contains(want, signalWindowOmissionMarker) {
					t.Fatal("fixture did not exercise representative sampling")
				}
			}
			if !reflect.DeepEqual(provider.texts, []string{want}) {
				t.Fatal("dispatcher or local scorer changed the selected input view")
			}
			if len(results.SignalErrors) != 0 || !reflect.DeepEqual(results.MatchedComplexityRules, []string{"scope:hard"}) {
				t.Fatalf("valid local score was lost: matches=%v errors=%v", results.MatchedComplexityRules, results.SignalErrors)
			}
		})
	}
}

type complexityRemoteInputRecorder struct{ texts []string }

func (p *complexityRemoteInputRecorder) Score(_ context.Context, text string) (float64, error) {
	p.texts = append(p.texts, text)
	return 0.9, nil
}

func (p *complexityRemoteInputRecorder) Classify(_ context.Context, text string) (SequenceClassificationResult, error) {
	p.texts = append(p.texts, text)
	return SequenceClassificationResult{Probabilities: []float32{0.9, 0.05, 0.05}}, nil
}

func TestEmbeddingFullContextDoesNotChangeRemoteComplexityInput(t *testing.T) {
	full := strings.Repeat("neutral context detail ", 400)
	for _, contract := range []string{config.RemoteClassifierContractScore, config.RemoteClassifierContractLabelDistribution} {
		t.Run(contract, func(t *testing.T) {
			cfg := remoteComplexityConfig(contract)
			cfg.EmbeddingConfig.ModelType = "mmbert"
			cfg.EmbeddingConfig.FullContext = true
			remote := &complexityRemoteInputRecorder{}
			classifier := &Classifier{Config: cfg}
			if contract == config.RemoteClassifierContractScore {
				classifier.complexityScoreBackend = remote
			} else {
				classifier.complexityLabelBackend = remote
			}
			results := classifier.EvaluateAllSignalsWithContext(full, full, full, nil, nil, false, true,
				"", nil, ConversationFacts{}, "")
			if !reflect.DeepEqual(remote.texts, []string{textForRoutingSignal(config.SignalTypeComplexity, full)}) {
				t.Fatal("embedding input policy changed an independent remote scorer")
			}
			if len(results.SignalErrors) != 0 || len(results.MatchedComplexityRules) != 1 {
				t.Fatal("remote scorer no longer produced a valid verdict")
			}
		})
	}
}

func TestComplexityFullContextPreservesProviderInputLimit(t *testing.T) {
	classifier, provider := newComplexityInputClassifier(t, true, 512)
	full := strings.Repeat("neutral context detail ", 400)
	results := classifier.EvaluateAllSignalsWithContext(full, full, full, nil, nil, false, true,
		"", nil, ConversationFacts{}, "")
	if !reflect.DeepEqual(provider.texts, []string{full}) || provider.limits.DeploymentTokens != 512 {
		t.Fatal("full context rewrote the request or provider budget")
	}
	if len(results.MatchedComplexityRules) != 0 || len(results.SignalValues) != 0 {
		t.Fatal("rejected input became a successful partial complexity score")
	}
	for _, verdict := range ComplexityVerdictLabels {
		if results.SignalErrors["complexity:scope:"+verdict] != complexityEvaluationFailedCode {
			t.Fatal("provider rejection lost the existing unknown signal contract")
		}
	}
	_, err := classifier.classifyComplexity(context.Background(), full, "", nil)
	if !errors.Is(err, binding.ErrInputLimit) {
		t.Fatalf("provider input-limit cause was not preserved: %v", err)
	}
}
