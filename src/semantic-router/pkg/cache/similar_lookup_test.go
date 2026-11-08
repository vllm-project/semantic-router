//go:build !windows

package cache

import (
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
