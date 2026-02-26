package memory

import (
	"math"
	"sort"
	"strings"
	"unicode"
)

// memBM25Doc is one document in the memory BM25 index.
type memBM25Doc struct {
	id     string
	terms  map[string]int
	length int
}

// MemBM25Index is a BM25 inverted index over memory content.
type MemBM25Index struct {
	docs  []memBM25Doc
	df    map[string]int
	avgDL float64
	n     int
}

// BuildMemBM25Index builds a BM25 index over the given id->content pairs.
func BuildMemBM25Index(docs map[string]string) *MemBM25Index {
	idx := &MemBM25Index{df: make(map[string]int)}
	totalLen := 0
	for id, content := range docs {
		tokens := memTokenize(content)
		tf := make(map[string]int, len(tokens))
		for _, t := range tokens {
			tf[t]++
		}
		idx.docs = append(idx.docs, memBM25Doc{id: id, terms: tf, length: len(tokens)})
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
func (idx *MemBM25Index) Score(query string, k1, b float64) map[string]float64 {
	if idx.n == 0 {
		return nil
	}
	queryTerms := memTokenize(query)
	if len(queryTerms) == 0 {
		return nil
	}
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
			scores[doc.id] = score
		}
	}
	return scores
}

// MemNgramIndex stores pre-computed character n-gram sets for Jaccard similarity.
type MemNgramIndex struct {
	ngrams map[string]map[string]struct{}
	n      int
}

// BuildMemNgramIndex builds an n-gram index over id->content pairs.
func BuildMemNgramIndex(docs map[string]string, n int) *MemNgramIndex {
	idx := &MemNgramIndex{
		ngrams: make(map[string]map[string]struct{}, len(docs)),
		n:      n,
	}
	for id, content := range docs {
		idx.ngrams[id] = memCharNgrams(strings.ToLower(content), n)
	}
	return idx
}

// Score returns Jaccard similarity in [0,1] for all docs against the query.
func (idx *MemNgramIndex) Score(query string) map[string]float64 {
	if len(idx.ngrams) == 0 {
		return nil
	}
	queryGrams := memCharNgrams(strings.ToLower(query), idx.n)
	if len(queryGrams) == 0 {
		return nil
	}
	scores := make(map[string]float64)
	for id, docGrams := range idx.ngrams {
		sim := memJaccard(queryGrams, docGrams)
		if sim > 0 {
			scores[id] = sim
		}
	}
	return scores
}

// MemHybridScorer holds pre-built BM25 and n-gram indexes for a set of memories
// and produces fused scores for queries.
type MemHybridScorer struct {
	bm25  *MemBM25Index
	ngram *MemNgramIndex
	cfg   *MemoryHybridConfig
}

// BuildMemHybridScorer builds BM25 and n-gram indexes from id->content pairs.
func BuildMemHybridScorer(docs map[string]string, cfg *MemoryHybridConfig) *MemHybridScorer {
	cfg.ApplyDefaults()
	return &MemHybridScorer{
		bm25:  BuildMemBM25Index(docs),
		ngram: BuildMemNgramIndex(docs, cfg.NgramSize),
		cfg:   cfg,
	}
}

// FusedScore computes the hybrid score for a single memory given its cosine
// similarity and the query string. Returns the fused score.
func (s *MemHybridScorer) FusedScore(memID string, cosineSim float32, query string) float32 {
	bm25Scores := s.bm25.Score(query, s.cfg.BM25K1, s.cfg.BM25B)
	ngramScores := s.ngram.Score(query)
	return s.fuseOne(memID, cosineSim, bm25Scores, ngramScores)
}

// FusedScores computes hybrid scores for all given memories in batch.
// cosineScores maps memID -> cosine similarity.
// Returns memID -> fused score.
func (s *MemHybridScorer) FusedScores(cosineScores map[string]float32, query string) map[string]float32 {
	bm25Scores := s.bm25.Score(query, s.cfg.BM25K1, s.cfg.BM25B)
	ngramScores := s.ngram.Score(query)

	if s.cfg.Mode == "rrf" {
		return s.fuseRRFBatch(cosineScores, bm25Scores, ngramScores)
	}
	return s.fuseWeightedBatch(cosineScores, bm25Scores, ngramScores)
}

func (s *MemHybridScorer) fuseOne(memID string, cosineSim float32, bm25Scores map[string]float64, ngramScores map[string]float64) float32 {
	wV, wB, wN := s.cfg.NormalizedWeights()

	bm25Raw := bm25Scores[memID]
	ngramRaw := ngramScores[memID]

	// BM25 needs normalization; for single-item we cap at 1.0 via sigmoid-like mapping
	normBM25 := 0.0
	if bm25Raw > 0 {
		normBM25 = bm25Raw / (bm25Raw + 1.0) // soft cap to [0,1)
	}

	return float32(wV*float64(cosineSim) + wB*normBM25 + wN*ngramRaw)
}

func (s *MemHybridScorer) fuseWeightedBatch(cosineScores map[string]float32, bm25Scores, ngramScores map[string]float64) map[string]float32 {
	wV, wB, wN := s.cfg.NormalizedWeights()

	// Min-max normalize BM25 scores across the batch
	var bm25Min, bm25Max float64
	first := true
	for id := range cosineScores {
		v := bm25Scores[id]
		if v == 0 {
			continue
		}
		if first {
			bm25Min, bm25Max = v, v
			first = false
		} else {
			if v < bm25Min {
				bm25Min = v
			}
			if v > bm25Max {
				bm25Max = v
			}
		}
	}
	bm25Range := bm25Max - bm25Min

	result := make(map[string]float32, len(cosineScores))
	for id, cosine := range cosineScores {
		normBM25 := 0.0
		if bm25Range > 0 {
			normBM25 = (bm25Scores[id] - bm25Min) / bm25Range
		} else if bm25Scores[id] > 0 {
			normBM25 = 1.0
		}
		result[id] = float32(wV*float64(cosine) + wB*normBM25 + wN*ngramScores[id])
	}
	return result
}

func (s *MemHybridScorer) fuseRRFBatch(cosineScores map[string]float32, bm25Scores, ngramScores map[string]float64) map[string]float32 {
	kf := float64(s.cfg.RRFConstant)
	result := make(map[string]float32, len(cosineScores))

	addRanks := func(scores map[string]float64) {
		type entry struct {
			id    string
			score float64
		}
		sorted := make([]entry, 0, len(scores))
		for id, sc := range scores {
			if _, inSet := cosineScores[id]; inSet {
				sorted = append(sorted, entry{id, sc})
			}
		}
		sort.Slice(sorted, func(i, j int) bool { return sorted[i].score > sorted[j].score })
		for rank, e := range sorted {
			result[e.id] += float32(1.0 / (kf + float64(rank+1)))
		}
	}

	// Vector rank
	type vEntry struct {
		id    string
		score float32
	}
	vSorted := make([]vEntry, 0, len(cosineScores))
	for id, s := range cosineScores {
		vSorted = append(vSorted, vEntry{id, s})
	}
	sort.Slice(vSorted, func(i, j int) bool { return vSorted[i].score > vSorted[j].score })
	for rank, e := range vSorted {
		result[e.id] += float32(1.0 / (kf + float64(rank+1)))
	}

	addRanks(bm25Scores)
	addRanks(ngramScores)

	return result
}

// memTokenize splits text into lowercase tokens on non-alphanumeric boundaries.
func memTokenize(text string) []string {
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

// memCharNgrams generates the set of character n-grams from text.
func memCharNgrams(text string, n int) map[string]struct{} {
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

// memJaccard computes |A ∩ B| / |A ∪ B|.
func memJaccard(a, b map[string]struct{}) float64 {
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
