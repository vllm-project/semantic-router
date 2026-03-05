package promptcompression

import (
	"math"
	"sync"
)

// TextRankScorer implements the TextRank algorithm for sentence importance scoring.
//
// Reference: Mihalcea, R. and Tarau, P. (2004). "TextRank: Bringing Order into
// Text." Proceedings of EMNLP 2004. ACL Anthology W04-3252.
//
// The algorithm models the document as a graph where sentences are nodes and
// edges are weighted by lexical similarity (cosine of TF vectors). It then
// runs a PageRank-style iterative computation to propagate importance scores
// through the graph. Sentences that are similar to many other important
// sentences receive higher scores — capturing the centrality intuition that
// the most "representative" sentences are the most important.
type TextRankScorer struct {
	dampingFactor float64
	maxIterations int
	convergence   float64
}

// NewTextRankScorer creates a TextRank scorer with standard parameters.
// The damping factor of 0.85 follows the original PageRank paper
// (Brin & Page, 1998) and is the value used in Mihalcea & Tarau (2004).
func NewTextRankScorer() *TextRankScorer {
	return &TextRankScorer{
		dampingFactor: 0.85,
		maxIterations: 100,
		convergence:   1e-5,
	}
}

// float64SlicePool reduces GC pressure from the PageRank power iteration.
// Each iteration needs a temporary []float64 of length n; pooling avoids
// allocating it on every pass.
var float64SlicePool = sync.Pool{
	New: func() interface{} {
		s := make([]float64, 0, 128)
		return &s
	},
}

func getFloat64Slice(n int) []float64 {
	sp := float64SlicePool.Get().(*[]float64)
	s := *sp
	if cap(s) >= n {
		s = s[:n]
	} else {
		s = make([]float64, n)
	}
	for i := range s {
		s[i] = 0
	}
	return s
}

func putFloat64Slice(s []float64) {
	s = s[:0]
	float64SlicePool.Put(&s)
}

// ScoreSentences computes TextRank importance scores for each sentence.
// Input: a slice of token lists (one per sentence). Output: normalized scores in [0, 1].
//
// GC note: the adjacency matrix is stored as a flat []float64 (row-major) instead
// of [][]float64 to avoid n slice-header allocations and improve cache locality.
// TF vectors are pre-computed once and reused across all pairwise comparisons.
func (tr *TextRankScorer) ScoreSentences(sentenceTokens [][]string) []float64 {
	n := len(sentenceTokens)
	if n == 0 {
		return nil
	}
	if n == 1 {
		return []float64{1.0}
	}

	// Pre-compute TF vectors once (avoids re-allocating maps in every
	// cosineSimilarity call — the dominant allocation source for large n).
	tfVecs := make([]map[string]float64, n)
	for i, tokens := range sentenceTokens {
		tfVecs[i] = termFrequency(tokens)
	}

	// Flat adjacency matrix: weights[i*n+j] instead of weights[i][j].
	// Single allocation of n*n float64 vs n allocations of []float64.
	weights := make([]float64, n*n)
	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			sim := cosineSimilarityFromTF(tfVecs[i], tfVecs[j])
			weights[i*n+j] = sim
			weights[j*n+i] = sim
		}
	}

	outSum := make([]float64, n)
	for i := 0; i < n; i++ {
		row := weights[i*n : (i+1)*n]
		for _, w := range row {
			outSum[i] += w
		}
	}

	scores := make([]float64, n)
	for i := range scores {
		scores[i] = 1.0 / float64(n)
	}

	d := tr.dampingFactor
	base := (1.0 - d) / float64(n)

	// Power iteration — reuse a pooled buffer for newScores to avoid
	// allocating a fresh []float64 on every iteration.
	newScores := getFloat64Slice(n)
	defer putFloat64Slice(newScores)

	for iter := 0; iter < tr.maxIterations; iter++ {
		maxDelta := 0.0

		for i := 0; i < n; i++ {
			var sum float64
			for j := 0; j < n; j++ {
				if i == j || outSum[j] == 0 {
					continue
				}
				sum += (weights[j*n+i] / outSum[j]) * scores[j]
			}
			newScores[i] = base + d*sum
			delta := math.Abs(newScores[i] - scores[i])
			if delta > maxDelta {
				maxDelta = delta
			}
		}

		// Swap instead of allocate: scores, newScores = newScores, scores
		scores, newScores = newScores, scores
		if maxDelta < tr.convergence {
			break
		}
		// Zero newScores for next iteration
		for i := range newScores {
			newScores[i] = 0
		}
	}

	// Normalize to [0, 1]
	maxScore := 0.0
	for _, s := range scores {
		if s > maxScore {
			maxScore = s
		}
	}
	if maxScore > 0 {
		for i := range scores {
			scores[i] /= maxScore
		}
	}

	return scores
}

// cosineSimilarityFromTF computes cosine similarity from pre-computed TF maps.
// Avoids the map allocation that cosineSimilarity would do per call.
func cosineSimilarityFromTF(tfA, tfB map[string]float64) float64 {
	if len(tfA) == 0 || len(tfB) == 0 {
		return 0
	}

	var dot, normA, normB float64
	for term, fa := range tfA {
		normA += fa * fa
		if fb, ok := tfB[term]; ok {
			dot += fa * fb
		}
	}
	for _, fb := range tfB {
		normB += fb * fb
	}

	denom := math.Sqrt(normA) * math.Sqrt(normB)
	if denom == 0 {
		return 0
	}
	return dot / denom
}

// cosineSimilarity computes cosine similarity between two token bags.
// Kept for test compatibility; the hot path uses cosineSimilarityFromTF.
func cosineSimilarity(a, b []string) float64 {
	if len(a) == 0 || len(b) == 0 {
		return 0
	}
	return cosineSimilarityFromTF(termFrequency(a), termFrequency(b))
}

func termFrequency(tokens []string) map[string]float64 {
	tf := make(map[string]float64, len(tokens))
	for _, t := range tokens {
		tf[t]++
	}
	n := float64(len(tokens))
	for k := range tf {
		tf[k] /= n
	}
	return tf
}
