package memory

import (
	"math"
	"testing"
)

func TestGenerateEmbeddingUsesDeterministicTestMode(t *testing.T) {
	t.Setenv(deterministicEmbeddingsEnv, "1")

	cfg := EmbeddingConfig{Model: EmbeddingModelMMBERT, Dimension: 384}
	got, err := GenerateEmbedding("My dog is a golden retriever named Max", cfg)
	if err != nil {
		t.Fatalf("GenerateEmbedding() error = %v", err)
	}
	if len(got) != 384 {
		t.Fatalf("GenerateEmbedding() len = %d, want 384", len(got))
	}

	again, err := GenerateEmbedding("My dog is a golden retriever named Max", cfg)
	if err != nil {
		t.Fatalf("GenerateEmbedding() repeat error = %v", err)
	}
	if score := cosine(got, again); math.Abs(score-1) > 1e-6 {
		t.Fatalf("deterministic embedding changed between calls")
	}
}

func TestDeterministicEmbeddingKeepsRelatedMemorySearchAboveThreshold(t *testing.T) {
	t.Setenv(deterministicEmbeddingsEnv, "1")

	cfg := EmbeddingConfig{Model: EmbeddingModelMMBERT, Dimension: 384}
	stored, err := GenerateEmbedding("My dog's name is Max and he is a golden retriever", cfg)
	if err != nil {
		t.Fatalf("GenerateEmbedding(stored) error = %v", err)
	}
	related, err := GenerateEmbedding("What is my dog's name and what breed is he?", cfg)
	if err != nil {
		t.Fatalf("GenerateEmbedding(related) error = %v", err)
	}
	unrelated, err := GenerateEmbedding("Tell me about quarterly revenue forecasts", cfg)
	if err != nil {
		t.Fatalf("GenerateEmbedding(unrelated) error = %v", err)
	}

	relatedScore := cosine(stored, related)
	unrelatedScore := cosine(stored, unrelated)
	if relatedScore <= 0.45 {
		t.Fatalf("related cosine = %.3f, want above memory threshold", relatedScore)
	}
	if relatedScore >= 0.99 {
		t.Fatalf("related cosine = %.3f, want below high threshold override", relatedScore)
	}
	if unrelatedScore >= relatedScore {
		t.Fatalf("unrelated cosine = %.3f, want below related %.3f", unrelatedScore, relatedScore)
	}
}

func TestDeterministicEmbeddingDoesNotApplySemanticRules(t *testing.T) {
	t.Setenv(deterministicEmbeddingsEnv, "1")

	cfg := EmbeddingConfig{Model: EmbeddingModelMMBERT, Dimension: 4096}
	dog, err := GenerateEmbedding("dog", cfg)
	if err != nil {
		t.Fatalf("GenerateEmbedding(dog) error = %v", err)
	}
	pet, err := GenerateEmbedding("pet", cfg)
	if err != nil {
		t.Fatalf("GenerateEmbedding(pet) error = %v", err)
	}
	if score := cosine(dog, pet); score != 0 {
		t.Fatalf("dog/pet cosine = %.3f, want no semantic aliasing", score)
	}
}

func cosine(a, b []float32) float64 {
	if len(a) != len(b) {
		return 0
	}
	var dot, normA, normB float64
	for i := range a {
		av := float64(a[i])
		bv := float64(b[i])
		dot += av * bv
		normA += av * av
		normB += bv * bv
	}
	if normA == 0 || normB == 0 {
		return 0
	}
	return dot / (math.Sqrt(normA) * math.Sqrt(normB))
}
