package promptcompression

import (
	"math"
)

// TFIDFScorer computes TF-IDF-based information density scores for sentences.
//
// This implements a token-level information measure inspired by Selective Context
// (Li et al. 2023, "Compressing Context to Enhance Inference Efficiency of Large
// Language Models", EMNLP 2023, arXiv:2310.06201). Selective Context uses
// self-information from a causal LM; we approximate this with TF-IDF which
// captures the same intuition: rare, informative tokens have high inverse
// document frequency and therefore high self-information.
//
// IDF(t) = log(N / df(t)) is proportional to the self-information of a token
// under a unigram corpus model: I(t) = -log P(t) ≈ log(N/df(t)).
type TFIDFScorer struct {
	docFreq  map[string]int
	numDocs  int
	idfCache map[string]float64
}

// NewTFIDFScorer builds document-frequency statistics from a set of sentence token lists.
//
// GC note: pre-sizes maps based on estimated vocabulary size to avoid
// repeated map growth for large documents. The "seen" set uses a reusable
// map that is cleared between sentences instead of allocating a new one.
func NewTFIDFScorer(sentenceTokens [][]string) *TFIDFScorer {
	totalTokens := 0
	for _, tokens := range sentenceTokens {
		totalTokens += len(tokens)
	}
	// Rough estimate: unique vocabulary ≈ sqrt(total tokens) for natural language (Heaps' law)
	vocabEstimate := totalTokens
	if vocabEstimate > 256 {
		vocabEstimate = int(math.Sqrt(float64(totalTokens)) * 3)
	}

	s := &TFIDFScorer{
		docFreq:  make(map[string]int, vocabEstimate),
		numDocs:  len(sentenceTokens),
		idfCache: make(map[string]float64, vocabEstimate),
	}

	// Reuse a single "seen" map across sentences to avoid per-sentence allocation.
	seen := make(map[string]bool, 64)
	for _, tokens := range sentenceTokens {
		for k := range seen {
			delete(seen, k)
		}
		for _, t := range tokens {
			if !seen[t] {
				s.docFreq[t]++
				seen[t] = true
			}
		}
	}

	return s
}

// IDF returns the inverse document frequency of a term.
func (s *TFIDFScorer) IDF(term string) float64 {
	if v, ok := s.idfCache[term]; ok {
		return v
	}
	df, ok := s.docFreq[term]
	if !ok || df == 0 {
		v := math.Log(float64(s.numDocs + 1))
		s.idfCache[term] = v
		return v
	}
	v := math.Log(float64(s.numDocs+1) / float64(df+1))
	s.idfCache[term] = v
	return v
}

// ScoreSentence returns the mean TF-IDF score for a sentence's tokens.
// Higher scores indicate sentences with more informative (rarer) content.
func (s *TFIDFScorer) ScoreSentence(tokens []string) float64 {
	if len(tokens) == 0 {
		return 0
	}

	tf := make(map[string]int, len(tokens))
	for _, t := range tokens {
		tf[t]++
	}

	var sum float64
	n := float64(len(tokens))
	for term, count := range tf {
		sum += (float64(count) / n) * s.IDF(term)
	}

	return sum
}

// ScoreSentenceWithTF returns the mean TF-IDF score using a pre-computed
// TF vector (map[term]frequency), avoiding a per-call map allocation.
func (s *TFIDFScorer) ScoreSentenceWithTF(tf map[string]float64) float64 {
	if len(tf) == 0 {
		return 0
	}
	var sum float64
	for term, freq := range tf {
		sum += freq * s.IDF(term)
	}
	return sum
}
