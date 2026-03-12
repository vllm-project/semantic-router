package cache

import (
	"math"
	"testing"
)

func TestScopeEmbeddingToUserReturnsOriginalWithoutUserID(t *testing.T) {
	embedding := []float32{3, 4, -2, 1}

	scoped := scopeEmbeddingToUser(embedding, "")

	if len(scoped) != len(embedding) {
		t.Fatalf("expected scoped embedding length %d, got %d", len(embedding), len(scoped))
	}
	for i := range embedding {
		if scoped[i] != embedding[i] {
			t.Fatalf("expected unchanged embedding at index %d, got %f want %f", i, scoped[i], embedding[i])
		}
	}
}

func TestScopeEmbeddingToUserIsStableForSameUser(t *testing.T) {
	embedding := testUserScopeEmbedding(64)

	first := scopeEmbeddingToUser(embedding, "user-a")
	second := scopeEmbeddingToUser(embedding, "user-a")

	if len(first) != len(second) {
		t.Fatalf("expected equal lengths, got %d and %d", len(first), len(second))
	}
	for i := range first {
		if math.Abs(float64(first[i]-second[i])) > 1e-6 {
			t.Fatalf("expected deterministic scoped embedding at index %d, diff=%f", i, first[i]-second[i])
		}
	}
}

func TestScopeEmbeddingToUserSeparatesDifferentUsers(t *testing.T) {
	embedding := testUserScopeEmbedding(128)

	userA := scopeEmbeddingToUser(embedding, "user-a")
	userB := scopeEmbeddingToUser(embedding, "user-b")
	similarity := testDotProduct(userA, userB)

	if similarity >= 0.8 {
		t.Fatalf("expected cross-user similarity below cache threshold, got %f", similarity)
	}
}

func TestScopeEmbeddingToUserSeparatesScopedAndUnscopedQueries(t *testing.T) {
	embedding := testUserScopeEmbedding(128)

	unscoped := normalizeEmbedding(embedding)
	scoped := scopeEmbeddingToUser(embedding, "user-a")
	similarity := testDotProduct(unscoped, scoped)

	if similarity >= 0.8 {
		t.Fatalf("expected scoped and unscoped similarity below cache threshold, got %f", similarity)
	}
}

func testUserScopeEmbedding(size int) []float32 {
	embedding := make([]float32, size)
	for i := range embedding {
		embedding[i] = float32((i%11)-5) / 5
	}
	return embedding
}

func testDotProduct(a, b []float32) float32 {
	limit := len(a)
	if len(b) < limit {
		limit = len(b)
	}

	var total float32
	for i := 0; i < limit; i++ {
		total += a[i] * b[i]
	}
	return total
}
