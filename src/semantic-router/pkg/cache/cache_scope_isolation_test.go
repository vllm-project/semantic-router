//go:build !windows && cgo

package cache

import (
	"strings"
	"testing"
)

func TestMilvusActiveEntryFilterIncludesEscapedModelPartition(t *testing.T) {
	filter := milvusActiveEntryFilterExpr(`privacy::model\"one`)
	if !strings.Contains(filter, `model == "privacy::model\\\"one"`) {
		t.Fatalf("model partition missing or unescaped in Milvus filter: %q", filter)
	}
	if !strings.Contains(filter, `response_body != ""`) {
		t.Fatalf("active-entry predicate missing from Milvus filter: %q", filter)
	}
}

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

// TestSearchEnforcesUserScope proves the hard scope gate drops a different
// user's entry during search even when its embedding is the nearest neighbor.
// Without it, a long identical query lands within the similarity threshold and
// one user receives another user's cached response (cross-tenant leak, observed
// live at similarity ~0.91 with threshold 0.8).
//
// It runs over BOTH in-memory search paths — linear scan and the HNSW index —
// because the gate is duplicated across scanLinearForSimilarity and
// scanHNSWCandidates and must hold identically on each. Embeddings are injected
// directly (the embedding model is a process boundary mocked here) so the test
// is deterministic and needs no loaded model; it drives the search through
// runFindSimilarEmbeddingSearch, which dispatches to the configured path.
func TestSearchEnforcesUserScope(t *testing.T) {
	tests := []struct {
		name    string
		useHNSW bool
	}{
		{name: "linear"},
		{name: "hnsw", useHNSW: true},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			assertSearchEnforcesUserScope(t, test.useHNSW)
		})
	}
}

func assertSearchEnforcesUserScope(t *testing.T, useHNSW bool) {
	t.Helper()

	const base = "a sufficiently long question whose embedding dominates the short scope prefix"
	const model = "model-a"
	emb := []float32{1, 0, 0}
	aliceResponse := []byte(`{"choices":[{"message":{"content":"alice-only"}}]}`)

	aliceScope := CacheScopeNamespaceOf(ScopeQueryToUser(base, "alice"))
	bobScope := CacheScopeNamespaceOf(ScopeQueryToUser(base, "bob"))

	c := NewInMemoryCache(InMemoryCacheOptions{
		Enabled:             true,
		SimilarityThreshold: 0.8,
		MaxEntries:          100,
		TTLSeconds:          0, // entries never expire in this test
		UseHNSW:             useHNSW,
		EmbeddingModel:      "mmbert",
	})
	defer c.Close()

	c.entries = append(c.entries, CacheEntry{
		RequestID:    "alice-1",
		Model:        model,
		Query:        ScopeQueryToUser(base, "alice"),
		ResponseBody: aliceResponse,
		Embedding:    emb,
	})
	if useHNSW {
		// Entries appended directly bypass the HNSW graph; mark it stale
		// so the search path rebuilds and actually sees alice's entry
		// (mirrors what AddEntry does in production).
		c.hnswNeedsRebuild = true
	}

	// Bob (different scope), IDENTICAL embedding → must NOT match Alice.
	if idx, _, _, _, _ := c.runFindSimilarEmbeddingSearch(emb, model, bobScope); idx != -1 {
		t.Fatalf("cross-tenant leak: bob matched alice's entry (idx=%d)", idx)
	}

	// Alice (same scope), same embedding → must match and return her response.
	idx, entry, _, _, _ := c.runFindSimilarEmbeddingSearch(emb, model, aliceScope)
	if idx != 0 {
		t.Fatalf("same-user lookup must hit, got idx=%d", idx)
	}
	if string(entry.ResponseBody) != string(aliceResponse) {
		t.Fatalf("same-user hit returned the wrong response body: %s", entry.ResponseBody)
	}

	// Anonymous (unscoped) lookup → must NOT match a scoped entry.
	if idx, _, _, _, _ := c.runFindSimilarEmbeddingSearch(emb, model, ""); idx != -1 {
		t.Fatalf("unscoped lookup must not match a scoped entry, idx=%d", idx)
	}

	if idx, _, _, _, _ := c.runFindSimilarEmbeddingSearch(emb, "model-b", aliceScope); idx != -1 {
		t.Fatalf("a different model partition must not match, idx=%d", idx)
	}
}
