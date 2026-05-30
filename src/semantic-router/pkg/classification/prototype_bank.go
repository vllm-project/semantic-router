package classification

import (
	"sort"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

type prototypeExample struct {
	Key       string
	Text      string
	Embedding []float32
}

type prototypeRepresentative struct {
	Key           string
	Text          string
	Embedding     []float32
	ClusterSize   int
	AvgSimilarity float64
}

type prototypeBank struct {
	config     config.PrototypeScoringConfig
	prototypes []prototypeRepresentative
}

func newPrototypeBank(examples []prototypeExample, cfg config.PrototypeScoringConfig) *prototypeBank {
	resolved := cfg.WithDefaults()
	deduped := dedupePrototypeExamples(examples)
	if len(deduped) == 0 {
		return &prototypeBank{config: resolved}
	}

	if !resolved.IsEnabled() {
		return &prototypeBank{
			config:     resolved,
			prototypes: uncompressedPrototypeRepresentatives(deduped),
		}
	}

	similarityMatrix := buildPrototypeSimilarityMatrix(deduped)
	clusters := clusterPrototypeExamples(deduped, similarityMatrix, float64(resolved.ClusterSimilarityThreshold))
	prototypes := make([]prototypeRepresentative, 0, len(clusters))
	for _, cluster := range clusters {
		prototypes = append(prototypes, selectPrototypeMedoid(deduped, cluster, similarityMatrix))
	}

	sortPrototypeRepresentatives(prototypes)
	if resolved.MaxPrototypes > 0 && len(prototypes) > resolved.MaxPrototypes {
		prototypes = prototypes[:resolved.MaxPrototypes]
	}

	return &prototypeBank{config: resolved, prototypes: prototypes}
}

func uncompressedPrototypeRepresentatives(examples []prototypeExample) []prototypeRepresentative {
	prototypes := make([]prototypeRepresentative, 0, len(examples))
	for _, example := range examples {
		prototypes = append(prototypes, prototypeRepresentative{
			Key:           example.Key,
			Text:          example.Text,
			Embedding:     example.Embedding,
			ClusterSize:   1,
			AvgSimilarity: 1,
		})
	}
	return prototypes
}

func sortPrototypeRepresentatives(prototypes []prototypeRepresentative) {
	sort.Slice(prototypes, func(i, j int) bool {
		if prototypes[i].ClusterSize != prototypes[j].ClusterSize {
			return prototypes[i].ClusterSize > prototypes[j].ClusterSize
		}
		if prototypes[i].AvgSimilarity != prototypes[j].AvgSimilarity {
			return prototypes[i].AvgSimilarity > prototypes[j].AvgSimilarity
		}
		return prototypes[i].Text < prototypes[j].Text
	})
}

func (b *prototypeBank) representatives() []prototypeRepresentative {
	if b == nil || len(b.prototypes) == 0 {
		return nil
	}
	representatives := make([]prototypeRepresentative, len(b.prototypes))
	copy(representatives, b.prototypes)
	return representatives
}
