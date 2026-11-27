package cache

import "sort"

// SimilarityBounds bound the cached entries a semantic lookup returns: each
// at least MinSimilarity similar to the query, and at most TopK of them.
type SimilarityBounds struct {
	MinSimilarity float32
	TopK          int
}

// fetch is how many nearest neighbours a backend asks its index for: its
// configured depth, and at least the lookup's TopK.
func (b SimilarityBounds) fetch(configured int) int {
	return max(configured, b.TopK)
}

// SimilarEntries are the cached entries a semantic lookup returned, most
// similar first, each with its cached request in Query. Closest is the best
// similarity the lookup saw, returned or not; a miss reports it on the debug
// and Replay surfaces.
type SimilarEntries struct {
	Hits    []LookupResult
	Closest float32
}

// similarHits collects a lookup's hits within its bounds and the closest
// similarity it saw.
type similarHits struct {
	bounds  SimilarityBounds
	hits    []LookupResult
	closest float32
	seen    bool
}

func (h *similarHits) saw(similarity float32) {
	if !h.seen || similarity > h.closest {
		h.closest, h.seen = similarity, true
	}
}

func (h *similarHits) add(hit LookupResult) {
	h.hits = append(h.hits, hit)
}

// entries orders the hits by similarity, ties in the order they came, and
// keeps the TopK best.
func (h *similarHits) entries() SimilarEntries {
	sort.SliceStable(h.hits, func(i, j int) bool { return h.hits[i].Similarity > h.hits[j].Similarity })
	if h.bounds.TopK > 0 && len(h.hits) > h.bounds.TopK {
		h.hits = h.hits[:h.bounds.TopK]
	}
	return SimilarEntries{Hits: h.hits, Closest: h.closest}
}

// bestSimilar is a lookup's most similar hit, or a miss that carries the
// closest similarity it saw.
func bestSimilar(entries SimilarEntries, err error) (LookupResult, error) {
	if err != nil {
		return LookupResult{}, err
	}
	if len(entries.Hits) == 0 {
		return LookupResult{Similarity: entries.Closest}, nil
	}
	return entries.Hits[0], nil
}
