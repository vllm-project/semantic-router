//go:build !windows && cgo

package cache

import (
	"testing"
	"time"
)

func TestCacheScopeNamespaceOfAndSameScope(t *testing.T) {
	base := "explain mitosis versus meiosis in eukaryotic cells in great detail"
	aliceQ := ScopeQueryToUser(base, "alice")
	bobQ := ScopeQueryToUser(base, "bob")

	if CacheScopeNamespaceOf(aliceQ) == "" {
		t.Fatal("scoped query must expose a namespace")
	}
	if got := CacheScopeNamespaceOf(base); got != "" {
		t.Fatalf("unscoped query must have empty namespace, got %q", got)
	}
	if SameCacheScope(aliceQ, bobQ) {
		t.Fatal("different users must not share a cache scope")
	}
	if !SameCacheScope(aliceQ, ScopeQueryToUser(base, "alice")) {
		t.Fatal("same user must share a cache scope")
	}
	if SameCacheScope(aliceQ, base) {
		t.Fatal("a scoped query must not match an unscoped one")
	}
}

// The hard scope gate must drop a different user's entry during search even
// when its embedding is the nearest neighbor. Without it, a long identical
// query lands within the similarity threshold and one user receives another
// user's cached response (cross-tenant leak, observed live at similarity
// ~0.91 with threshold 0.8).
func TestLinearScanEnforcesUserScope(t *testing.T) {
	c := NewInMemoryCache(InMemoryCacheOptions{
		Enabled:             true,
		SimilarityThreshold: 0.8,
		MaxEntries:          100,
		TTLSeconds:          0, // entries never expire in this test
		UseHNSW:             false,
		EmbeddingModel:      "mmbert",
	})

	base := "a sufficiently long question whose embedding dominates the short scope prefix"
	emb := []float32{1, 0, 0}
	c.entries = append(c.entries, CacheEntry{
		RequestID:    "alice-1",
		Query:        ScopeQueryToUser(base, "alice"),
		ResponseBody: []byte(`{"choices":[{"message":{"content":"alice-only"}}]}`),
		Embedding:    emb,
	})

	now := time.Now()

	// Bob (different scope), IDENTICAL embedding → must NOT match Alice.
	bobScope := CacheScopeNamespaceOf(ScopeQueryToUser(base, "bob"))
	if idx, _, _, _ := c.scanLinearForSimilarity(emb, bobScope, now); idx != -1 {
		t.Fatalf("cross-tenant leak: bob matched alice's entry (idx=%d)", idx)
	}

	// Alice (same scope), same embedding → must match.
	aliceScope := CacheScopeNamespaceOf(ScopeQueryToUser(base, "alice"))
	if idx, _, _, _ := c.scanLinearForSimilarity(emb, aliceScope, now); idx != 0 {
		t.Fatalf("same-user lookup must hit, got idx=%d", idx)
	}

	// Anonymous (unscoped) lookup → must NOT match a scoped entry.
	if idx, _, _, _ := c.scanLinearForSimilarity(emb, "", now); idx != -1 {
		t.Fatalf("unscoped lookup must not match a scoped entry, idx=%d", idx)
	}
}
