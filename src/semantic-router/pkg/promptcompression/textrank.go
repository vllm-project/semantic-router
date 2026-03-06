package promptcompression

import (
	"math"
	"runtime"
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

// adjacencyPool holds flat []float64 matrices (n*n) for TextRank.
// At maxSentences=500, this is 500*500*8 = 2MB — large enough that
// recycling avoids measurable GC pressure on the hot path.
var adjacencyPool = sync.Pool{
	New: func() interface{} {
		s := make([]float64, 0, 256*256)
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

func getAdjacencyMatrix(n int) []float64 {
	size := n * n
	sp := adjacencyPool.Get().(*[]float64)
	s := *sp
	if cap(s) >= size {
		s = s[:size]
	} else {
		s = make([]float64, size)
	}
	for i := range s {
		s[i] = 0
	}
	return s
}

func putAdjacencyMatrix(s []float64) {
	s = s[:0]
	adjacencyPool.Put(&s)
}

// ScoreSentences computes TextRank importance scores for each sentence.
// Input: a slice of token lists (one per sentence). Output: normalized scores in [0, 1].
// This is the backward-compatible entry point that builds TF vectors internally.
func (tr *TextRankScorer) ScoreSentences(sentenceTokens [][]string) []float64 {
	n := len(sentenceTokens)
	if n == 0 {
		return nil
	}
	if n == 1 {
		return []float64{1.0}
	}
	tfVecs := make([]map[string]float64, n)
	for i, tokens := range sentenceTokens {
		tfVecs[i] = termFrequency(tokens)
	}
	return tr.scoreSentencesFromTF(n, tfVecs)
}

// ScoreSentencesWithTF computes TextRank scores using pre-computed TF vectors,
// avoiding duplicate map allocations when the caller already has TF data.
func (tr *TextRankScorer) ScoreSentencesWithTF(tfVecs []map[string]float64) []float64 {
	n := len(tfVecs)
	if n == 0 {
		return nil
	}
	if n == 1 {
		return []float64{1.0}
	}
	return tr.scoreSentencesFromTF(n, tfVecs)
}

// scoreSentencesFromTF is the shared implementation.
//
// GC note: the adjacency matrix is stored as a flat []float64 (row-major) instead
// of [][]float64 to avoid n slice-header allocations and improve cache locality.
// The matrix is obtained from a sync.Pool to avoid re-allocating n*n*8 bytes per call.
func (tr *TextRankScorer) scoreSentencesFromTF(n int, tfVecs []map[string]float64) []float64 {
	weights := getAdjacencyMatrix(n)
	defer putAdjacencyMatrix(weights)

	// Pre-compute L2 norms so the pairwise loop avoids redundant iteration.
	norms := make([]float64, n)
	for i, tf := range tfVecs {
		var sq float64
		for _, v := range tf {
			sq += v * v
		}
		norms[i] = math.Sqrt(sq)
	}

	// Parallelize pairwise cosine computation for large n. The threshold
	// avoids goroutine overhead when n is small. Each goroutine writes to
	// non-overlapping (i,j)/(j,i) cells so no synchronization is needed
	// beyond the WaitGroup.
	const parallelThreshold = 64
	if n >= parallelThreshold {
		numWorkers := runtime.GOMAXPROCS(0)
		if numWorkers > n {
			numWorkers = n
		}
		var wg sync.WaitGroup
		wg.Add(numWorkers)
		chunkSize := (n + numWorkers - 1) / numWorkers
		for w := 0; w < numWorkers; w++ {
			lo := w * chunkSize
			hi := lo + chunkSize
			if hi > n {
				hi = n
			}
			go func(lo, hi int) {
				defer wg.Done()
				for i := lo; i < hi; i++ {
					for j := i + 1; j < n; j++ {
						sim := cosineSimilarityWithNorms(tfVecs[i], tfVecs[j], norms[i], norms[j])
						weights[i*n+j] = sim
						weights[j*n+i] = sim
					}
				}
			}(lo, hi)
		}
		wg.Wait()
	} else {
		for i := 0; i < n; i++ {
			for j := i + 1; j < n; j++ {
				sim := cosineSimilarityWithNorms(tfVecs[i], tfVecs[j], norms[i], norms[j])
				weights[i*n+j] = sim
				weights[j*n+i] = sim
			}
		}
	}

	outSum := make([]float64, n)
	for i := 0; i < n; i++ {
		row := weights[i*n : (i+1)*n]
		for _, w := range row {
			outSum[i] += w
		}
	}

	// Build the transition matrix T in-place: T[i][j] = weights[j][i] / outSum[j].
	// Since weights is symmetric, weights[j][i] == weights[i][j], so
	// T[i][j] = weights[i][j] / outSum[j]. The power iteration inner loop then
	// becomes a row-major dot product (cache-friendly) instead of column-access.
	for i := 0; i < n; i++ {
		row := weights[i*n : (i+1)*n]
		for j := 0; j < n; j++ {
			if i == j || outSum[j] == 0 {
				row[j] = 0
			} else {
				row[j] /= outSum[j]
			}
		}
	}

	scores := make([]float64, n)
	for i := range scores {
		scores[i] = 1.0 / float64(n)
	}

	d := tr.dampingFactor
	base := (1.0 - d) / float64(n)

	newScores := getFloat64Slice(n)
	defer putFloat64Slice(newScores)

	for iter := 0; iter < tr.maxIterations; iter++ {
		maxDelta := 0.0

		for i := 0; i < n; i++ {
			var sum float64
			row := weights[i*n : (i+1)*n]
			for j := 0; j < n; j++ {
				sum += row[j] * scores[j]
			}
			newScores[i] = base + d*sum
			delta := math.Abs(newScores[i] - scores[i])
			if delta > maxDelta {
				maxDelta = delta
			}
		}

		scores, newScores = newScores, scores
		if maxDelta < tr.convergence {
			break
		}
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

// cosineSimilarityWithNorms computes cosine similarity using pre-computed L2 norms.
// Only the dot product requires map iteration; the denominator is norms[i]*norms[j].
// Iterates the smaller map for the dot product to minimize hash lookups.
func cosineSimilarityWithNorms(tfA, tfB map[string]float64, normA, normB float64) float64 {
	denom := normA * normB
	if denom == 0 {
		return 0
	}
	small, large := tfA, tfB
	if len(tfA) > len(tfB) {
		small, large = tfB, tfA
	}
	var dot float64
	for term, fs := range small {
		if fl, ok := large[term]; ok {
			dot += fs * fl
		}
	}
	return dot / denom
}

// cosineSimilarityFromTF computes cosine similarity from pre-computed TF maps.
// Iterates the smaller map for the dot product to minimize hash lookups.
func cosineSimilarityFromTF(tfA, tfB map[string]float64) float64 {
	if len(tfA) == 0 || len(tfB) == 0 {
		return 0
	}

	small, large := tfA, tfB
	if len(tfA) > len(tfB) {
		small, large = tfB, tfA
	}

	var dot, normSmall, normLarge float64
	for term, fs := range small {
		normSmall += fs * fs
		if fl, ok := large[term]; ok {
			dot += fs * fl
		}
	}
	for _, fl := range large {
		normLarge += fl * fl
	}

	denom := math.Sqrt(normSmall) * math.Sqrt(normLarge)
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
