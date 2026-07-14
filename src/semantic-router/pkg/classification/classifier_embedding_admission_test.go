package classification

import (
	"context"
	"errors"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
)

type admissionRemoteProvider struct{}

func (admissionRemoteProvider) Embed(context.Context, string) ([]float32, error) {
	return []float32{1}, nil
}

func (admissionRemoteProvider) EmbedBatch(context.Context, []string) ([][]float32, error) {
	return [][]float32{{1}}, nil
}
func (admissionRemoteProvider) Dimension() int { return 1 }
func (admissionRemoteProvider) Backend() string {
	return config.EmbeddingBackendOpenAICompatible
}

func TestUsesLocalNativeEmbeddingsUsesRuntimeProvidersAndImageRules(t *testing.T) {
	t.Setenv("EMBEDDING_BACKEND_OVERRIDE", "")
	local := &Classifier{keywordEmbeddingClassifier: &EmbeddingClassifier{
		rulesByModality: map[config.QueryModality][]config.EmbeddingRule{
			config.QueryModalityText: {{Name: "local-text"}},
		},
	}}
	if !local.UsesLocalNativeEmbeddings(false) {
		t.Fatal("local text embedding classifier was not admitted")
	}

	remote := &Classifier{keywordEmbeddingClassifier: &EmbeddingClassifier{
		backend:  config.EmbeddingBackendOpenAICompatible,
		provider: admissionRemoteProvider{},
		rulesByModality: map[config.QueryModality][]config.EmbeddingRule{
			config.QueryModalityText: {{Name: "remote-text"}},
		},
	}}
	if remote.UsesLocalNativeEmbeddings(false) {
		t.Fatal("provider-backed text-only classifier consumed local admission")
	}
	remote.keywordEmbeddingClassifier.rulesByModality[config.QueryModalityImage] = []config.EmbeddingRule{{Name: "local-image"}}
	if !remote.UsesLocalNativeEmbeddings(true) {
		t.Fatal("local multimodal image rule bypassed admission")
	}
}

func TestUsesLocalNativeEmbeddingsHonorsOverrideForEveryProviderBackedClassifier(t *testing.T) {
	provider := admissionRemoteProvider{}
	classifiers := map[string]*Classifier{
		"embedding": {
			keywordEmbeddingClassifier: &EmbeddingClassifier{
				backend:  config.EmbeddingBackendOpenAICompatible,
				provider: provider,
				rulesByModality: map[config.QueryModality][]config.EmbeddingRule{
					config.QueryModalityText: {{Name: "remote-text"}},
				},
			},
		},
		"reask": {
			reaskClassifier: &ReaskClassifier{provider: provider},
		},
		"complexity": {
			complexityClassifier: &ComplexityClassifier{provider: provider},
		},
		"preference": {
			preferenceClassifier: &PreferenceClassifier{
				useContrastive: true,
				contrastive:    &ContrastivePreferenceClassifier{provider: provider},
			},
		},
		"contrastive jailbreak": {
			contrastiveJailbreakClassifiers: map[string]*ContrastiveJailbreakClassifier{
				"jailbreak": {provider: provider},
			},
		},
		"knowledge base": {
			kbClassifiers: map[string]*KnowledgeBaseClassifier{
				"kb": {provider: provider},
			},
		},
	}

	for name, classifier := range classifiers {
		t.Run(name, func(t *testing.T) {
			t.Setenv("EMBEDDING_BACKEND_OVERRIDE", "")
			if classifier.UsesLocalNativeEmbeddings(false) {
				t.Fatal("remote backend unexpectedly consumed native admission")
			}
			for _, override := range []string{config.EmbeddingBackendCandle, config.EmbeddingBackendOpenVINO} {
				t.Run(override, func(t *testing.T) {
					t.Setenv("EMBEDDING_BACKEND_OVERRIDE", override)
					if !classifier.UsesLocalNativeEmbeddings(false) {
						t.Fatalf("provider-backed classifier bypassed native admission under %q override", override)
					}
				})
			}
		})
	}
}

func TestUsesLocalNativeEmbeddingsRemoteOverrideDoesNotPretendToUseNative(t *testing.T) {
	t.Setenv("EMBEDDING_BACKEND_OVERRIDE", config.EmbeddingBackendOpenAICompatible)
	classifier := &Classifier{reaskClassifier: &ReaskClassifier{}}
	if classifier.UsesLocalNativeEmbeddings(false) {
		t.Fatal("remote override unexpectedly consumed native admission")
	}
	if _, err := classifier.reaskClassifier.embedText("must not fall back"); err == nil {
		t.Fatal("remote override without a provider silently fell back to native inference")
	}
}

func TestEvaluateAllSignalsPropagatesContentSafeImageFailure(t *testing.T) {
	original := getMultiModalImageEmbedding
	getMultiModalImageEmbedding = func(string, int) ([]float32, error) {
		return nil, errors.New("private model path")
	}
	t.Cleanup(func() { getMultiModalImageEmbedding = original })

	classifier := classifierWithEmbeddingOnly(t, []config.EmbeddingRule{{
		Name:          "image-rule",
		Candidates:    []string{"safe image"},
		QueryModality: config.QueryModalityImage,
	}}, config.EmbeddingModelTypeQwen3)
	classifier.Config = &config.RouterConfig{IntelligentRouting: config.IntelligentRouting{
		Decisions: []config.Decision{{
			Name:  "image-route",
			Rules: config.RuleCombination{Type: config.SignalTypeEmbedding, Name: "image-rule"},
		}},
	}}

	results, err := classifier.EvaluateAllSignalsWithHeaders(
		"", "", "", nil, nil, false, nil, false, "data:image/png;base64,AA==",
	)
	if results != nil {
		t.Fatalf("failed image evaluation returned results: %+v", results)
	}
	if !errors.Is(err, ErrImageSignalEvaluation) || errors.Is(err, ErrInvalidImageSignalInput) {
		t.Fatalf("image evaluation error = %v", err)
	}
	if strings.Contains(err.Error(), "private model path") {
		t.Fatalf("public image evaluation error exposed internal detail: %v", err)
	}
}

func TestEvaluateAllSignalsFailsClosedOnTextEmbeddingFailure(t *testing.T) {
	original := getEmbeddingWithModelType
	getEmbeddingWithModelType = func(string, string, int) (*candle_binding.EmbeddingOutput, error) {
		return nil, errors.New("private remote backend detail")
	}
	t.Cleanup(func() { getEmbeddingWithModelType = original })

	classifier := classifierWithEmbeddingOnly(t, []config.EmbeddingRule{{
		Name:       "secure-route",
		Candidates: []string{"safe candidate"},
	}}, config.EmbeddingModelTypeQwen3)
	classifier.Config = &config.RouterConfig{IntelligentRouting: config.IntelligentRouting{
		Decisions: []config.Decision{{
			Name:  "secure",
			Rules: config.RuleCombination{Type: config.SignalTypeEmbedding, Name: "secure-route"},
		}},
	}}

	results, err := classifier.EvaluateAllSignalsWithHeaders(
		"request", "request", "request", nil, nil, false, nil, false, "",
	)
	if results != nil || !errors.Is(err, ErrTextSignalEvaluation) {
		t.Fatalf("text backend failure = results %+v err %v", results, err)
	}
	if strings.Contains(err.Error(), "private remote backend detail") {
		t.Fatalf("typed error exposed backend detail: %v", err)
	}
}

func TestContrastiveJailbreakEmbeddingFailureIsFailClosed(t *testing.T) {
	provider, err := embedding.NewFuncProvider(
		config.EmbeddingBackendOpenAICompatible,
		2,
		func(context.Context, string) ([]float32, error) {
			return nil, errors.New("private security provider detail")
		},
	)
	if err != nil {
		t.Fatalf("NewFuncProvider: %v", err)
	}
	rule := config.JailbreakRule{Name: "security", Method: "contrastive", Threshold: 0.1}
	classifier := &Classifier{contrastiveJailbreakClassifiers: map[string]*ContrastiveJailbreakClassifier{
		rule.Name: {modelType: config.EmbeddingModelTypeRemote, backend: config.EmbeddingBackendOpenAICompatible, provider: provider},
	}}
	results := newSignalResultsForTest()
	var mu sync.Mutex
	classifier.evaluateContrastiveJailbreakRule(rule, []string{"unsafe request"}, time.Now(), results, &mu)

	if err := results.EvaluationError(); !errors.Is(err, ErrTextSignalEvaluation) {
		t.Fatalf("security evaluation error = %v", err)
	}
	if len(results.MatchedJailbreakRules) != 0 {
		t.Fatalf("failed security backend produced matches: %v", results.MatchedJailbreakRules)
	}
}

func TestEvaluateEmbeddingSignalClassifiesInvalidImageFailure(t *testing.T) {
	original := getMultiModalImageEmbedding
	getMultiModalImageEmbedding = func(string, int) ([]float32, error) {
		return nil, candle_binding.ErrInvalidImageInput
	}
	t.Cleanup(func() { getMultiModalImageEmbedding = original })

	classifier := classifierWithEmbeddingOnly(t, []config.EmbeddingRule{{
		Name:          "image-rule",
		Candidates:    []string{"safe image"},
		QueryModality: config.QueryModalityImage,
	}}, config.EmbeddingModelTypeQwen3)
	results := newSignalResultsForTest()
	var mu sync.Mutex
	classifier.evaluateEmbeddingSignal(results, &mu, "", "data:image/png;base64,AA==", nil)

	err := results.ImageEvaluationError()
	if !errors.Is(err, ErrImageSignalEvaluation) || !errors.Is(err, ErrInvalidImageSignalInput) {
		t.Fatalf("invalid image evaluation error = %v", err)
	}
}

func TestComplexityImageOnlySkipsTextEmbeddingAndPropagatesImageFailure(t *testing.T) {
	originalText := getEmbeddingWithModelType
	originalImage := getMultiModalImageEmbedding
	var textCalls atomic.Int32
	getEmbeddingWithModelType = func(string, string, int) (*candle_binding.EmbeddingOutput, error) {
		textCalls.Add(1)
		return nil, errors.New("text embedding must not run")
	}
	getMultiModalImageEmbedding = func(string, int) ([]float32, error) {
		return nil, errors.New("image runtime unavailable")
	}
	t.Cleanup(func() {
		getEmbeddingWithModelType = originalText
		getMultiModalImageEmbedding = originalImage
	})

	classifier := &ComplexityClassifier{hasImageCandidates: true}
	_, err := classifier.loadQueryEmbeddingsCached("", "data:image/png;base64,AA==", nil)
	if !errors.Is(err, ErrImageSignalEvaluation) {
		t.Fatalf("complexity image-only error = %v", err)
	}
	if textCalls.Load() != 0 {
		t.Fatalf("image-only complexity performed %d text embeddings", textCalls.Load())
	}
}
