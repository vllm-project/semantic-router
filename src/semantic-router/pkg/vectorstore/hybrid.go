/*
Copyright 2025 vLLM Semantic Router.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package vectorstore

import (
	"context"
	"fmt"
	"math"
	"sort"
	"strings"
	"unicode"
)

// HybridSearcher is an optional interface that backends can implement
// to support hybrid search combining vector similarity with BM25 and
// n-gram text matching.
type HybridSearcher interface {
	HybridSearch(ctx context.Context, vectorStoreID string,
		query string, queryEmbedding []float32,
		topK int, threshold float32,
		filter map[string]interface{},
		config *HybridSearchConfig,
	) ([]SearchResult, error)
}

// HybridSearchConfig configures hybrid search behavior.
type HybridSearchConfig struct {
	// Mode selects the score fusion method: "weighted" (default) or "rrf".
	Mode string `json:"mode,omitempty"`

	// Weights for weighted mode. Relative values; normalized to sum to 1.0.
	VectorWeight float64 `json:"vector_weight,omitempty"` // default 0.7
	BM25Weight   float64 `json:"bm25_weight,omitempty"`   // default 0.2
	NgramWeight  float64 `json:"ngram_weight,omitempty"`  // default 0.1

	// RRFConstant for RRF mode (default 60).
	RRFConstant int `json:"rrf_constant,omitempty"`

	// BM25K1 controls term frequency saturation (default 1.2).
	BM25K1 float64 `json:"bm25_k1,omitempty"`
	// BM25B controls document length normalization (default 0.75).
	BM25B float64 `json:"bm25_b,omitempty"`

	// NgramSize is the character n-gram size for Jaccard similarity (default 3).
	NgramSize int `json:"ngram_size,omitempty"`
}

func (c *HybridSearchConfig) applyDefaults() {
	if c.Mode == "" {
		c.Mode = "weighted"
	}
	if c.VectorWeight == 0 && c.BM25Weight == 0 && c.NgramWeight == 0 {
		c.VectorWeight = 0.7
		c.BM25Weight = 0.2
		c.NgramWeight = 0.1
	}
	if c.RRFConstant <= 0 {
		c.RRFConstant = 60
	}
	if c.BM25K1 == 0 {
		c.BM25K1 = 1.2
	}
	if c.BM25B == 0 {
		c.BM25B = 0.75
	}
	if c.NgramSize <= 0 {
		c.NgramSize = 3
	}
}

// normalizedWeights returns vector, bm25, ngram weights that sum to 1.
func (c *HybridSearchConfig) normalizedWeights() (float64, float64, float64) {
	total := c.VectorWeight + c.BM25Weight + c.NgramWeight
	if total == 0 {
		return 1.0 / 3, 1.0 / 3, 1.0 / 3
	}
	return c.VectorWeight / total, c.BM25Weight / total, c.NgramWeight / total
}

// ---------------------------------------------------------------------------
// BM25 Index
// ---------------------------------------------------------------------------

type bm25Doc struct {
	chunkID string
	terms   map[string]int // term -> frequency
	length  int
}

// BM25Index is a pre-built inverted index that supports BM25 scoring
// with configurable k1/b parameters at query time.
type BM25Index struct {
	docs  []bm25Doc
	df    map[string]int // term -> document frequency
	avgDL float64
	n     int // total number of documents
}

// NewBM25Index builds a BM25 inverted index over the given chunks.
func NewBM25Index(chunks map[string]EmbeddedChunk) *BM25Index {
	idx := &BM25Index{
		df: make(map[string]int),
	}

	totalLen := 0
	for id, chunk := range chunks {
		tokens := tokenize(chunk.Content)
		tf := make(map[string]int, len(tokens))
		for _, t := range tokens {
			tf[t]++
		}
		idx.docs = append(idx.docs, bm25Doc{
			chunkID: id,
			terms:   tf,
			length:  len(tokens),
		})
		totalLen += len(tokens)
	}

	idx.n = len(idx.docs)
	if idx.n > 0 {
		idx.avgDL = float64(totalLen) / float64(idx.n)
	}

	for _, doc := range idx.docs {
		seen := make(map[string]struct{}, len(doc.terms))
		for term := range doc.terms {
			if _, ok := seen[term]; !ok {
				idx.df[term]++
				seen[term] = struct{}{}
			}
		}
	}

	return idx
}

// Score returns BM25 scores for all documents against the query.
func (idx *BM25Index) Score(query string, k1, b float64) map[string]float64 {
	if idx.n == 0 {
		return nil
	}

	queryTerms := tokenize(query)
	if len(queryTerms) == 0 {
		return nil
	}

	// Pre-compute IDF for query terms only.
	idfCache := make(map[string]float64, len(queryTerms))
	for _, qt := range queryTerms {
		if _, ok := idfCache[qt]; ok {
			continue
		}
		n := float64(idx.df[qt])
		if n == 0 {
			idfCache[qt] = 0
			continue
		}
		idfCache[qt] = math.Log((float64(idx.n)-n+0.5)/(n+0.5) + 1)
	}

	scores := make(map[string]float64)
	for _, doc := range idx.docs {
		var score float64
		for _, qt := range queryTerms {
			idf := idfCache[qt]
			if idf == 0 {
				continue
			}
			tf := float64(doc.terms[qt])
			if tf == 0 {
				continue
			}
			num := tf * (k1 + 1)
			denom := tf + k1*(1-b+b*float64(doc.length)/idx.avgDL)
			score += idf * num / denom
		}
		if score > 0 {
			scores[doc.chunkID] = score
		}
	}

	return scores
}

// ---------------------------------------------------------------------------
// N-gram Index
// ---------------------------------------------------------------------------

// NgramIndex stores pre-computed character n-gram sets for Jaccard similarity.
type NgramIndex struct {
	chunkNgrams map[string]map[string]struct{}
	n           int
}

// NewNgramIndex builds an n-gram index over the given chunks.
func NewNgramIndex(chunks map[string]EmbeddedChunk, n int) *NgramIndex {
	idx := &NgramIndex{
		chunkNgrams: make(map[string]map[string]struct{}, len(chunks)),
		n:           n,
	}
	for id, chunk := range chunks {
		idx.chunkNgrams[id] = charNgrams(strings.ToLower(chunk.Content), n)
	}
	return idx
}

// Score returns Jaccard similarity in [0,1] for all chunks against the query.
func (idx *NgramIndex) Score(query string) map[string]float64 {
	if len(idx.chunkNgrams) == 0 {
		return nil
	}

	queryGrams := charNgrams(strings.ToLower(query), idx.n)
	if len(queryGrams) == 0 {
		return nil
	}

	scores := make(map[string]float64)
	for chunkID, docGrams := range idx.chunkNgrams {
		sim := jaccardSimilarity(queryGrams, docGrams)
		if sim > 0 {
			scores[chunkID] = sim
		}
	}
	return scores
}

// NgramN returns the n-gram size this index was built with.
func (idx *NgramIndex) NgramN() int {
	return idx.n
}

// ---------------------------------------------------------------------------
// Tokenizer
// ---------------------------------------------------------------------------

// tokenize splits text into lowercase tokens on non-alphanumeric boundaries.
func tokenize(text string) []string {
	var tokens []string
	var cur strings.Builder
	for _, r := range strings.ToLower(text) {
		if unicode.IsLetter(r) || unicode.IsDigit(r) {
			cur.WriteRune(r)
		} else if cur.Len() > 0 {
			tokens = append(tokens, cur.String())
			cur.Reset()
		}
	}
	if cur.Len() > 0 {
		tokens = append(tokens, cur.String())
	}
	return tokens
}

// ---------------------------------------------------------------------------
// N-gram helpers
// ---------------------------------------------------------------------------

// charNgrams generates the set of character n-grams from text.
func charNgrams(text string, n int) map[string]struct{} {
	runes := []rune(text)
	if len(runes) < n {
		result := make(map[string]struct{})
		if len(runes) > 0 {
			result[string(runes)] = struct{}{}
		}
		return result
	}
	result := make(map[string]struct{}, len(runes)-n+1)
	for i := 0; i <= len(runes)-n; i++ {
		result[string(runes[i:i+n])] = struct{}{}
	}
	return result
}

// jaccardSimilarity computes |A ∩ B| / |A ∪ B|.
func jaccardSimilarity(a, b map[string]struct{}) float64 {
	if len(a) == 0 && len(b) == 0 {
		return 0
	}
	intersection := 0
	for k := range a {
		if _, ok := b[k]; ok {
			intersection++
		}
	}
	union := len(a) + len(b) - intersection
	if union == 0 {
		return 0
	}
	return float64(intersection) / float64(union)
}

// ---------------------------------------------------------------------------
// Score Fusion
// ---------------------------------------------------------------------------

// fusedChunk holds per-retriever scores for a single chunk.
type fusedChunk struct {
	chunkID     string
	vectorScore float64
	bm25Score   float64
	ngramScore  float64
	finalScore  float64
}

// FuseScores combines scores from vector, BM25, and n-gram retrievers
// using either weighted combination or Reciprocal Rank Fusion (RRF).
// Returns fused results sorted by finalScore descending.
func FuseScores(
	vectorScores, bm25Scores, ngramScores map[string]float64,
	config *HybridSearchConfig,
) []fusedChunk {
	config.applyDefaults()

	all := make(map[string]*fusedChunk)
	for id, s := range vectorScores {
		all[id] = &fusedChunk{chunkID: id, vectorScore: s}
	}
	for id, s := range bm25Scores {
		if fc, ok := all[id]; ok {
			fc.bm25Score = s
		} else {
			all[id] = &fusedChunk{chunkID: id, bm25Score: s}
		}
	}
	for id, s := range ngramScores {
		if fc, ok := all[id]; ok {
			fc.ngramScore = s
		} else {
			all[id] = &fusedChunk{chunkID: id, ngramScore: s}
		}
	}

	if config.Mode == "rrf" {
		fuseRRF(all, vectorScores, bm25Scores, ngramScores, config.RRFConstant)
	} else {
		fuseWeighted(all, config)
	}

	results := make([]fusedChunk, 0, len(all))
	for _, fc := range all {
		results = append(results, *fc)
	}
	sort.Slice(results, func(i, j int) bool {
		return results[i].finalScore > results[j].finalScore
	})
	return results
}

// fuseWeighted computes weighted sum with min-max normalization for BM25.
// Vector scores (cosine) and n-gram scores (Jaccard) are already in [0,1].
func fuseWeighted(all map[string]*fusedChunk, config *HybridSearchConfig) {
	wV, wB, wN := config.normalizedWeights()

	// Min-max normalize BM25 scores (they are unbounded).
	var bm25Min, bm25Max float64
	first := true
	for _, fc := range all {
		if fc.bm25Score == 0 {
			continue
		}
		if first {
			bm25Min, bm25Max = fc.bm25Score, fc.bm25Score
			first = false
		} else {
			if fc.bm25Score < bm25Min {
				bm25Min = fc.bm25Score
			}
			if fc.bm25Score > bm25Max {
				bm25Max = fc.bm25Score
			}
		}
	}
	bm25Range := bm25Max - bm25Min

	for _, fc := range all {
		normBM25 := 0.0
		if bm25Range > 0 {
			normBM25 = (fc.bm25Score - bm25Min) / bm25Range
		} else if fc.bm25Score > 0 {
			normBM25 = 1.0
		}
		fc.finalScore = wV*fc.vectorScore + wB*normBM25 + wN*fc.ngramScore
	}
}

// fuseRRF applies Reciprocal Rank Fusion: score(d) = Σ_r 1/(k + rank_r(d)).
func fuseRRF(
	all map[string]*fusedChunk,
	vectorScores, bm25Scores, ngramScores map[string]float64,
	k int,
) {
	kf := float64(k)

	addRanks := func(scores map[string]float64) {
		if len(scores) == 0 {
			return
		}
		type entry struct {
			id    string
			score float64
		}
		sorted := make([]entry, 0, len(scores))
		for id, s := range scores {
			sorted = append(sorted, entry{id, s})
		}
		sort.Slice(sorted, func(i, j int) bool {
			return sorted[i].score > sorted[j].score
		})
		for rank, e := range sorted {
			if fc, ok := all[e.id]; ok {
				fc.finalScore += 1.0 / (kf + float64(rank+1))
			}
		}
	}

	addRanks(vectorScores)
	addRanks(bm25Scores)
	addRanks(ngramScores)
}

// ---------------------------------------------------------------------------
// Generic Hybrid Rerank (works with any VectorStoreBackend)
// ---------------------------------------------------------------------------

// rerankCandidateMultiplier controls how many extra candidates to fetch from
// the base vector search so that text-based re-ranking has a richer pool.
const rerankCandidateMultiplier = 4

// GenericHybridRerank performs hybrid search on any backend by re-ranking
// the results of the backend's existing vector Search method with BM25 and
// n-gram text scores. This is the fallback path for backends that do not
// implement HybridSearcher natively (e.g. Milvus, Llama Stack).
//
// Flow:
//  1. Call backend.Search with an expanded topK (4x) and threshold=0 to get
//     a broad candidate set from vector similarity.
//  2. Build ephemeral BM25 and n-gram indexes over the candidate contents.
//  3. Fuse the three score signals using the configured mode (weighted / RRF).
//  4. Apply the caller's threshold and topK to the fused results.
func GenericHybridRerank(
	ctx context.Context,
	backend VectorStoreBackend,
	vectorStoreID string,
	query string,
	queryEmbedding []float32,
	topK int,
	threshold float32,
	filter map[string]interface{},
	config *HybridSearchConfig,
) ([]SearchResult, error) {
	if config == nil {
		config = &HybridSearchConfig{}
	}
	config.applyDefaults()

	// Fetch an expanded candidate set from the base vector search.
	expandedTopK := topK * rerankCandidateMultiplier
	if expandedTopK < 50 {
		expandedTopK = 50
	}

	vectorResults, err := backend.Search(ctx, vectorStoreID, queryEmbedding,
		expandedTopK, 0, filter)
	if err != nil {
		return nil, fmt.Errorf("hybrid rerank: vector search failed: %w", err)
	}

	if len(vectorResults) == 0 {
		return vectorResults, nil
	}

	// Build pseudo-chunks keyed by a stable index ID so the BM25 / n-gram
	// indexes can reference the same candidates as the vector results.
	pseudoChunks := make(map[string]EmbeddedChunk, len(vectorResults))
	vectorScores := make(map[string]float64, len(vectorResults))
	idxToKey := make([]string, len(vectorResults))

	for i, r := range vectorResults {
		key := fmt.Sprintf("_rr_%d", i)
		idxToKey[i] = key
		pseudoChunks[key] = EmbeddedChunk{ID: key, Content: r.Content}
		vectorScores[key] = r.Score
	}

	// Score with BM25.
	bm25Idx := NewBM25Index(pseudoChunks)
	bm25Scores := bm25Idx.Score(query, config.BM25K1, config.BM25B)

	// Score with n-gram.
	ngramIdx := NewNgramIndex(pseudoChunks, config.NgramSize)
	ngramScores := ngramIdx.Score(query)

	// Fuse.
	fused := FuseScores(vectorScores, bm25Scores, ngramScores, config)

	// Map fused chunk keys back to original results.
	keyToIdx := make(map[string]int, len(idxToKey))
	for i, k := range idxToKey {
		keyToIdx[k] = i
	}

	results := make([]SearchResult, 0, topK)
	for _, fc := range fused {
		if fc.finalScore < float64(threshold) {
			continue
		}
		origIdx, ok := keyToIdx[fc.chunkID]
		if !ok {
			continue
		}
		orig := vectorResults[origIdx]
		vs := fc.vectorScore
		bs := fc.bm25Score
		ns := fc.ngramScore
		results = append(results, SearchResult{
			FileID:      orig.FileID,
			Filename:    orig.Filename,
			Content:     orig.Content,
			Score:       fc.finalScore,
			ChunkIndex:  orig.ChunkIndex,
			VectorScore: &vs,
			BM25Score:   &bs,
			NgramScore:  &ns,
		})
		if topK > 0 && len(results) >= topK {
			break
		}
	}

	return results, nil
}
