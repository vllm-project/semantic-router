//go:build !windows

package cache

import (
	"context"
	"reflect"
	"testing"
	"time"
)

// A semantic lookup returns every entry within its bounds, most similar
// first, at most TopK, each with the cached request it answered; the closest
// similarity counts the entries it did not return.
func TestInMemoryLookupReturnsTheTopKMostSimilarEntries(t *testing.T) {
	for _, useHNSW := range []bool{false, true} {
		c := NewInMemoryCache(InMemoryCacheOptions{Enabled: true, MaxEntries: 16, UseHNSW: useHNSW, EmbeddingDeployment: "vela-embedding"})
		c.entries = append(c.entries,
			CacheEntry{RequestID: "a", Query: "what is the refund window", ResponseBody: []byte("A"), Embedding: []float32{0.95, 0.31}},
			CacheEntry{RequestID: "b", Query: "how long can I return items", ResponseBody: []byte("B"), Embedding: []float32{0.99, 0.14}},
			CacheEntry{RequestID: "c", Query: "refund policy for orders", ResponseBody: []byte("C"), Embedding: []float32{0.85, 0.53}},
			CacheEntry{RequestID: "d", Query: "where is my parcel", ResponseBody: []byte("D"), Embedding: []float32{0.2, 0.98}},
		)
		c.hnswNeedsRebuild = useHNSW
		query := "refund window for my order"
		bounds := SimilarityBounds{MinSimilarity: 0.8, TopK: 2}
		search := c.runFindSimilarEmbeddingSearch([]float32{1, 0}, "", query, bounds, "")
		entries := c.finishFindSimilarSearch(time.Now(), "", query, bounds, search)
		var bodies, queries []string
		for _, hit := range entries.Hits {
			bodies, queries = append(bodies, string(hit.ResponseBody)), append(queries, hit.Query)
		}
		if !reflect.DeepEqual(bodies, []string{"B", "A"}) || queries[0] != "how long can I return items" {
			t.Fatalf("hnsw=%v: hits %v %v, want B then A", useHNSW, bodies, queries)
		}
		if entries.Closest < 0.98 {
			t.Fatalf("hnsw=%v: closest %v", useHNSW, entries.Closest)
		}
		c.Close()
	}
}

func TestSimilarHitsKeepTheBestTopK(t *testing.T) {
	ranked := similarHits{bounds: SimilarityBounds{TopK: 2}}
	for _, similarity := range []float32{0.81, 0.97, 0.88} {
		ranked.saw(similarity)
		ranked.add(LookupResult{Found: true, Similarity: similarity})
	}
	ranked.saw(0.99)
	entries := ranked.entries()
	if len(entries.Hits) != 2 || entries.Hits[0].Similarity != 0.97 || entries.Hits[1].Similarity != 0.88 || entries.Closest != 0.99 {
		t.Fatalf("entries %+v", entries)
	}
	if miss, _ := bestSimilar(SimilarEntries{Closest: 0.7}, nil); miss.Found || miss.Similarity != 0.7 {
		t.Fatalf("a miss carries the closest similarity: %+v", miss)
	}
}

// A semantic lookup recalls top_k entries for the caller to choose from, but
// only the entry it serves counts as used, and no lookup moves an entry's
// expiry: an entry lives the TTL its write set, as on Redis.
func TestInMemorySemanticLookupMarksOnlyTheServedEntry(t *testing.T) {
	backend := NewInMemoryCache(InMemoryCacheOptions{Enabled: true, MaxEntries: 16, TTLSeconds: 60, EmbeddingProvider: cacheTestEmbeddingProvider()})
	t.Cleanup(func() { _ = backend.Close() })
	service := NewResponseCacheService(NewLegacyBackendAdapter(backend, InMemoryCacheType), DefaultResponseCacheServiceOptions())
	identity := CacheIdentity{Partition: CachePartition{RequestModel: "m"}}
	for _, query := range []string{"what is the refund window", "how long can I return items"} {
		identity.SemanticQuery = query
		write := CacheWrite{Identity: identity, RequestID: query, ResponseBody: []byte(query), TTL: TTL(time.Minute)}
		if err := service.StoreSemantic(context.Background(), write); err != nil {
			t.Fatal(err)
		}
	}
	written := map[string]time.Time{}
	for _, entry := range backend.entries {
		written[entry.Query] = entry.ExpiresAt
	}

	identity.SemanticQuery = "refund window for my order"
	recall, err := service.LookupSemantic(context.Background(), SemanticLookup{Identity: identity, Bounds: SimilarityBounds{MinSimilarity: -1, TopK: 2}})
	if err != nil || len(recall.Candidates) != 2 {
		t.Fatalf("recalled %d entries, err %v", len(recall.Candidates), err)
	}
	served := recall.Candidates[1]
	service.RecordSemantic(identity, served, nil, time.Now())

	for _, entry := range backend.entries {
		wantHits := int64(0)
		if entry.Query == served.Query {
			wantHits = 1
		}
		if entry.HitCount != wantHits || !entry.ExpiresAt.Equal(written[entry.Query]) {
			t.Errorf("%q: hits %d expiry moved %v, want hits %d and the written expiry",
				entry.Query, entry.HitCount, entry.ExpiresAt.Sub(written[entry.Query]), wantHits)
		}
	}
}
