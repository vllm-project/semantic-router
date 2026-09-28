package classification

import (
	"context"
	"encoding/base64"
	"fmt"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
)

// Every test owns its callbacks; concurrent classifiers never share an override.
type testEmbeddingProvider struct {
	text  func(context.Context, string, embedding.Options) ([]float32, error)
	image func(context.Context, []byte, int) ([]float32, error)
}

func (p *testEmbeddingProvider) Embed(ctx context.Context, text string) ([]float32, error) {
	return p.EmbedWithOptions(ctx, text, embedding.Options{})
}

func (p *testEmbeddingProvider) EmbedWithOptions(ctx context.Context, text string, options embedding.Options) ([]float32, error) {
	if p.text == nil {
		return nil, fmt.Errorf("unexpected text embedding: %q", text)
	}
	return p.text(ctx, text, options)
}

func (p *testEmbeddingProvider) EmbedBatch(ctx context.Context, texts []string) ([][]float32, error) {
	result := make([][]float32, len(texts))
	for i, text := range texts {
		value, err := p.Embed(ctx, text)
		if err != nil {
			return nil, err
		}
		result[i] = value
	}
	return result, nil
}
func (p *testEmbeddingProvider) Dimension() int  { return 768 }
func (p *testEmbeddingProvider) Backend() string { return "synthetic" }
func (p *testEmbeddingProvider) EmbedImage(ctx context.Context, data []byte, dimension int) ([]float32, error) {
	if p.image == nil {
		return nil, fmt.Errorf("unexpected image embedding")
	}
	return p.image(ctx, data, dimension)
}

func newTestTextProvider(fn func(string) ([]float32, error)) *testEmbeddingProvider {
	return &testEmbeddingProvider{text: func(_ context.Context, text string, _ embedding.Options) ([]float32, error) { return fn(text) }}
}

func testImageURI(name string) string {
	return "data:image/png;base64," + base64.StdEncoding.EncodeToString([]byte(name))
}

func testImageURIs(names []string) []string {
	result := make([]string, len(names))
	for i, name := range names {
		result[i] = testImageURI(name)
	}
	return result
}

func TestEmbeddingConsumersRequireOwnedProviders(t *testing.T) {
	for name, infer := range map[string]func() ([]float32, error){
		"embedding": func() ([]float32, error) {
			return (&EmbeddingClassifier{}).computeEmbedding(context.Background(), "query", "multimodal")
		},
		"category":   func() ([]float32, error) { return (&KnowledgeBaseClassifier{}).embedText("query") },
		"preference": func() ([]float32, error) { return (&ContrastivePreferenceClassifier{}).embedText("query") },
		"jailbreak":  func() ([]float32, error) { return (&ContrastiveJailbreakClassifier{}).embedText("query") },
		"reask":      func() ([]float32, error) { return (&ReaskClassifier{}).embedText("query") },
		"complexity": func() ([]float32, error) {
			return (&ComplexityClassifier{}).computeCandidateEmbedding(complexityCandidateTask{candidate: "query"})
		},
	} {
		t.Run(name, func(t *testing.T) {
			if _, err := infer(); err == nil || !strings.Contains(err.Error(), "provider was not prepared") {
				t.Fatalf("missing owned provider did not fail explicitly: %v", err)
			}
		})
	}
}
