package classification

import (
	"sort"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

type prototypeBankScore struct {
	Score          float64
	Best           float64
	Support        float64
	PrototypeCount int
}

type prototypeScoreOptions struct {
	BestWeight float64
	TopM       int
}

func defaultPrototypeScoreOptions(cfg config.PrototypeScoringConfig) prototypeScoreOptions {
	resolved := cfg.WithDefaults()
	return prototypeScoreOptions{
		BestWeight: float64(resolved.BestWeight),
		TopM:       resolved.TopM,
	}
}

func (b *prototypeBank) score(queryEmbedding []float32, options prototypeScoreOptions) prototypeBankScore {
	if b == nil || len(b.prototypes) == 0 {
		return prototypeBankScore{}
	}

	similarities := make([]float64, 0, len(b.prototypes))
	for _, prototype := range b.prototypes {
		similarities = append(similarities, float64(cosineSimilarity(queryEmbedding, prototype.Embedding)))
	}
	sort.Slice(similarities, func(i, j int) bool { return similarities[i] > similarities[j] })

	best := similarities[0]
	topM := options.TopM
	if topM <= 0 || topM > len(similarities) {
		topM = len(similarities)
	}

	support := 0.0
	for _, similarity := range similarities[:topM] {
		support += similarity
	}
	support /= float64(topM)

	bestWeight := options.BestWeight
	if bestWeight < 0 {
		bestWeight = 0
	}
	if bestWeight > 1 {
		bestWeight = 1
	}

	return prototypeBankScore{
		Score:          bestWeight*best + (1-bestWeight)*support,
		Best:           best,
		Support:        support,
		PrototypeCount: len(b.prototypes),
	}
}
