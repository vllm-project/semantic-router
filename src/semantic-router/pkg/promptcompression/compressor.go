// Package promptcompression provides NLP-based prompt compression that reduces
// long prompts to a token budget while preserving classification and embedding
// fidelity. It uses no LLM inference — only classical NLP scoring.
//
// The compression pipeline combines four well-cited scoring signals:
//
//  1. TextRank (Mihalcea & Tarau, EMNLP 2004) — graph-based sentence importance
//     via PageRank over a cosine-similarity adjacency matrix.
//
//  2. Position weighting (Liu et al., "Lost in the Middle", TACL 2024,
//     arXiv:2307.03172) — U-shaped attention curve that upweights sentences at
//     the beginning and end of the prompt where transformer attention is strongest.
//
//  3. TF-IDF information density (inspired by Selective Context, Li et al.,
//     EMNLP 2023, arXiv:2310.06201) — sentences with rare, informative tokens
//     score higher, approximating token-level self-information without an LM.
//
//  4. Novelty scoring (adapted from Radev et al., IPM 2004; Carbonell &
//     Goldstein, SIGIR 1998) — inverse of centroid-based centrality. Sentences
//     whose TF vectors diverge from the document centroid score high, surfacing
//     outlier content (jailbreak prefixes, PII) without keyword lists.
//
// Sentences are ranked by a weighted combination of these four scores. The
// top-ranked sentences are selected to fit the token budget and reassembled in
// their original order to preserve discourse coherence.
//
// GC considerations: the pipeline is designed to minimize heap allocations for
// large messages (16K+ tokens). Key strategies:
//   - sync.Pool for TextRank power-iteration buffers
//   - Flat (row-major) adjacency matrix instead of [][]float64
//   - Pre-computed TF vectors reused across pairwise comparisons
//   - Map pre-sizing with Heaps' law vocabulary estimates
//   - Reused "seen" sets in TF-IDF document frequency computation
//   - Capacity-hinted slice allocations throughout
//   - selectSentences uses a bitset instead of map[int]bool
package promptcompression

import (
	"math"
	"sort"
	"strings"
)

// Config controls the compression behavior.
type Config struct {
	// MaxTokens is the target token budget. If the input is already within
	// budget, it is returned unchanged.
	MaxTokens int

	// TextRankWeight controls the contribution of TextRank content-importance
	// scores. Default: 0.20.
	TextRankWeight float64

	// PositionWeight controls the contribution of Lost-in-the-Middle position
	// scores. High weight preserves domain signals at prompt boundaries.
	// Default: 0.40.
	PositionWeight float64

	// TFIDFWeight controls the contribution of TF-IDF information density
	// scores. Default: 0.35.
	TFIDFWeight float64

	// NoveltyWeight controls the contribution of novelty (inverse centrality)
	// scores. Kept low to avoid displacing domain-representative content with
	// outlier sentences. Default: 0.05.
	NoveltyWeight float64

	// PositionDepth controls the amplitude of the U-shaped position curve.
	// 0 = flat (no position bias), 1 = maximum penalty for middle sentences.
	// Default: 0.5 (middle sentences get half the weight of edges).
	PositionDepth float64

	// PreserveFirstN always keeps the first N sentences regardless of score.
	// Motivated by the primacy effect in "Lost in the Middle": the opening
	// sentences provide critical framing context. Default: 3 (covers system
	// prompt, jailbreak prefixes, and initial PII in typical LLM API payloads).
	PreserveFirstN int

	// PreserveLastN always keeps the last N sentences regardless of score.
	// Motivated by the recency effect: the final sentences often contain
	// the user's actual request or question. Default: 2.
	PreserveLastN int
}

// DefaultConfig returns a Config with empirically reasonable defaults.
func DefaultConfig(maxTokens int) Config {
	return Config{
		MaxTokens:      maxTokens,
		TextRankWeight: 0.20,
		PositionWeight: 0.40,
		TFIDFWeight:    0.35,
		NoveltyWeight:  0.05,
		PositionDepth:  0.5,
		PreserveFirstN: 3,
		PreserveLastN:  2,
	}
}

// Result holds the compression output and diagnostic metadata.
type Result struct {
	// Compressed is the compressed text, sentences joined by spaces.
	Compressed string

	// OriginalTokens is the estimated token count of the input.
	OriginalTokens int

	// CompressedTokens is the estimated token count of the output.
	CompressedTokens int

	// Ratio is CompressedTokens / OriginalTokens.
	Ratio float64

	// SentenceScores holds the composite score for each original sentence,
	// useful for debugging and test assertions.
	SentenceScores []ScoredSentence

	// KeptIndices lists the original indices of sentences that were kept.
	KeptIndices []int
}

// ScoredSentence pairs a sentence with its component and composite scores.
type ScoredSentence struct {
	Index     int
	Text      string
	Tokens    int
	TextRank  float64
	Position  float64
	TFIDF     float64
	Novelty   float64
	Composite float64
}

// maxSentences caps the number of sentences fed into the TextRank O(n²)
// adjacency matrix. Beyond this, the matrix allocation (n*n*8 bytes) and
// pairwise cosine computation dominate latency and GC pressure. At 500
// sentences the matrix is 500*500*8 = 2MB, which fits comfortably in L2
// cache and avoids multi-MB GC sweeps on the hot path.
//
// When the sentence count exceeds this cap, we uniformly sample while
// always preserving the first and last sentences (for primacy/recency).
const maxSentences = 500

// Compress reduces the input text to fit within the configured token budget.
// If the input already fits, it is returned unchanged with Ratio = 1.0.
//
// The input text is never modified — Compress is a pure function that returns
// a new string assembled from selected original sentences. The caller must
// send the original (uncompressed) text to the upstream model; the compressed
// result is intended only for signal extraction / classification.
func Compress(text string, cfg Config) Result {
	originalTokens := CountTokensApprox(text)

	if cfg.MaxTokens <= 0 || originalTokens <= cfg.MaxTokens {
		return Result{
			Compressed:       text,
			OriginalTokens:   originalTokens,
			CompressedTokens: originalTokens,
			Ratio:            1.0,
		}
	}

	sentences := SplitSentences(text)
	if len(sentences) <= 1 {
		return Result{
			Compressed:       text,
			OriginalTokens:   originalTokens,
			CompressedTokens: originalTokens,
			Ratio:            1.0,
		}
	}

	// Cap sentence count to avoid O(n²) blowup in TextRank.
	// Uniform sampling preserves first + last for primacy/recency.
	if len(sentences) > maxSentences {
		sentences = sampleSentences(sentences, maxSentences)
	}

	n := len(sentences)
	sentTokens := make([][]string, n)
	sentTokenCounts := make([]int, n)
	for i, s := range sentences {
		sentTokens[i] = TokenizeWords(s)
		sentTokenCounts[i] = CountTokensApprox(s)
	}

	// --- Score computation ---
	normalizeWeights(&cfg)

	// Pre-compute TF vectors once — shared by TextRank, TF-IDF, and Novelty.
	tfVecs := make([]map[string]float64, n)
	for i, tokens := range sentTokens {
		tf := make(map[string]float64, len(tokens))
		for _, t := range tokens {
			tf[t]++
		}
		cnt := float64(len(tokens))
		for k := range tf {
			tf[k] /= cnt
		}
		tfVecs[i] = tf
	}

	textRankScores := NewTextRankScorer().ScoreSentencesWithTF(tfVecs)
	positionScores := PositionWeights(n, cfg.PositionDepth)
	tfidfScorer := NewTFIDFScorer(sentTokens)

	tfidfScores := make([]float64, n)
	for i := range sentences {
		tfidfScores[i] = tfidfScorer.ScoreSentenceWithTF(tfVecs[i])
	}
	normalizeSlice(tfidfScores)

	var noveltyScores []float64
	if cfg.NoveltyWeight > 0 {
		noveltyScorer := NewNoveltyScorer(tfVecs)
		noveltyScores = make([]float64, n)
		for i := range sentences {
			noveltyScores[i] = noveltyScorer.ScoreSentence(tfVecs[i])
		}
		normalizeSlice(noveltyScores)
	}

	scored := make([]ScoredSentence, n)
	for i := range sentences {
		var nov float64
		if noveltyScores != nil {
			nov = noveltyScores[i]
		}
		scored[i] = ScoredSentence{
			Index:    i,
			Text:     sentences[i],
			Tokens:   sentTokenCounts[i],
			TextRank: textRankScores[i],
			Position: positionScores[i],
			TFIDF:    tfidfScores[i],
			Novelty:  nov,
			Composite: cfg.TextRankWeight*textRankScores[i] +
				cfg.PositionWeight*positionScores[i] +
				cfg.TFIDFWeight*tfidfScores[i] +
				cfg.NoveltyWeight*nov,
		}
	}

	// --- Sentence selection ---
	kept := selectSentences(scored, sentTokenCounts, cfg)

	sort.Ints(kept)

	parts := make([]string, 0, len(kept))
	compressedTokens := 0
	for _, idx := range kept {
		parts = append(parts, sentences[idx])
		compressedTokens += sentTokenCounts[idx]
	}

	compressed := strings.Join(parts, " ")
	ratio := 0.0
	if originalTokens > 0 {
		ratio = float64(compressedTokens) / float64(originalTokens)
	}

	return Result{
		Compressed:       compressed,
		OriginalTokens:   originalTokens,
		CompressedTokens: compressedTokens,
		Ratio:            ratio,
		SentenceScores:   scored,
		KeptIndices:      kept,
	}
}

// selectSentences picks sentences to keep within the token budget.
// It first reserves mandatory first/last sentences, then greedily adds
// the highest-scoring remaining sentences.
//
// GC note: uses a flat bool slice (bitset) instead of map[int]bool, and
// pre-allocates the candidates slice with capacity n.
func selectSentences(scored []ScoredSentence, tokenCounts []int, cfg Config) []int {
	n := len(scored)
	kept := make([]bool, n) // bitset — no map overhead
	budget := cfg.MaxTokens
	usedTokens := 0
	keptCount := 0

	// Reserve first N sentences (primacy)
	for i := 0; i < cfg.PreserveFirstN && i < n; i++ {
		if !kept[i] && usedTokens+tokenCounts[i] <= budget {
			kept[i] = true
			usedTokens += tokenCounts[i]
			keptCount++
		}
	}

	// Reserve last N sentences (recency)
	for i := n - cfg.PreserveLastN; i < n; i++ {
		if i >= 0 && !kept[i] && usedTokens+tokenCounts[i] <= budget {
			kept[i] = true
			usedTokens += tokenCounts[i]
			keptCount++
		}
	}

	// Rank remaining sentences by composite score (descending)
	type candidate struct {
		index int
		score float64
	}
	candidates := make([]candidate, 0, n-keptCount)
	for i := range scored {
		if !kept[i] {
			candidates = append(candidates, candidate{i, scored[i].Composite})
		}
	}
	sort.Slice(candidates, func(a, b int) bool {
		return candidates[a].score > candidates[b].score
	})

	// Greedily add highest-scoring sentences that fit
	for _, c := range candidates {
		if usedTokens+tokenCounts[c.index] <= budget {
			kept[c.index] = true
			usedTokens += tokenCounts[c.index]
			keptCount++
		}
	}

	result := make([]int, 0, keptCount)
	for i, ok := range kept {
		if ok {
			result = append(result, i)
		}
	}
	return result
}

// sampleSentences uniformly selects n sentences from the input, always
// preserving the first and last sentences. Deterministic (no randomness)
// so results are reproducible.
func sampleSentences(sentences []string, n int) []string {
	total := len(sentences)
	if total <= n {
		return sentences
	}
	result := make([]string, 0, n)
	result = append(result, sentences[0])

	// Fill middle slots with uniformly spaced sentences
	middle := n - 2
	if middle > 0 {
		step := float64(total-2) / float64(middle)
		for i := 0; i < middle; i++ {
			idx := 1 + int(float64(i)*step+0.5)
			if idx >= total-1 {
				idx = total - 2
			}
			result = append(result, sentences[idx])
		}
	}

	result = append(result, sentences[total-1])
	return result
}

// normalizeWeights ensures the four score weights sum to 1.0.
func normalizeWeights(cfg *Config) {
	total := cfg.TextRankWeight + cfg.PositionWeight + cfg.TFIDFWeight + cfg.NoveltyWeight
	if total <= 0 {
		cfg.TextRankWeight = 0.25
		cfg.PositionWeight = 0.25
		cfg.TFIDFWeight = 0.25
		cfg.NoveltyWeight = 0.25
		return
	}
	cfg.TextRankWeight /= total
	cfg.PositionWeight /= total
	cfg.TFIDFWeight /= total
	cfg.NoveltyWeight /= total
}

// normalizeSlice scales values to [0, 1] by dividing by the max.
func normalizeSlice(vals []float64) {
	if len(vals) == 0 {
		return
	}
	maxVal := math.Inf(-1)
	for _, v := range vals {
		if v > maxVal {
			maxVal = v
		}
	}
	if maxVal <= 0 {
		return
	}
	for i := range vals {
		vals[i] /= maxVal
	}
}
