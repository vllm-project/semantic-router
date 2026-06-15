package promptcompression

import "math"

// NoveltyScorer scores sentences by how dissimilar they are from the document
// centroid. Outlier sentences -- those with vocabulary very different from the
// average -- receive high scores. This naturally surfaces safety-critical
// content (jailbreak prefixes, PII, anomalous instructions) that TextRank
// would suppress because it rewards centrality rather than uniqueness.
//
// The score is: novelty(s) = 1 - cosine(TF(s), centroid)
// where centroid = mean of all sentence TF vectors.
//
// This requires no pattern matching or keyword lists; it relies purely on the
// distributional observation that adversarial or sensitive content uses
// vocabulary distinct from the document's dominant topic.
type NoveltyScorer struct {
	centroid     map[string]float64
	centroidNorm float64 // pre-computed sqrt(sum(v^2)) of centroid
}

// NewNoveltyScorer builds a document centroid from pre-computed TF vectors.
// The centroid's L2 norm is cached so ScoreSentence avoids re-iterating
// the centroid map on every call.
func NewNoveltyScorer(tfVecs []map[string]float64) *NoveltyScorer {
	if len(tfVecs) == 0 {
		return &NoveltyScorer{centroid: make(map[string]float64)}
	}

	centroid := make(map[string]float64, len(tfVecs[0])*2)
	n := float64(len(tfVecs))

	for _, tf := range tfVecs {
		for term, freq := range tf {
			centroid[term] += freq / n
		}
	}

	var normSq float64
	for _, v := range centroid {
		normSq += v * v
	}

	return &NoveltyScorer{
		centroid:     centroid,
		centroidNorm: math.Sqrt(normSq),
	}
}

// ScoreSentence returns the novelty of a sentence: 1 - cosine(tf, centroid).
// Range [0, 1]. Higher = more dissimilar from the document average.
func (ns *NoveltyScorer) ScoreSentence(tf map[string]float64) float64 {
	if len(tf) == 0 || len(ns.centroid) == 0 {
		return 0
	}

	var dot, normA float64
	for term, fa := range tf {
		normA += fa * fa
		if fb, ok := ns.centroid[term]; ok {
			dot += fa * fb
		}
	}

	denom := math.Sqrt(normA) * ns.centroidNorm
	if denom == 0 {
		return 1.0
	}

	cosine := dot / denom
	return 1.0 - cosine
}
