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

func newPrototypeBank(examples []prototypeExample, cfg config.PrototypeScoringConfig) *prototypeBank {
	resolved := cfg.WithDefaults()
	deduped := dedupePrototypeExamples(examples)
	if len(deduped) == 0 {
		return &prototypeBank{config: resolved}
	}

	if !resolved.IsEnabled() {
		prototypes := make([]prototypeRepresentative, 0, len(deduped))
		for _, example := range deduped {
			prototypes = append(prototypes, prototypeRepresentative{
				Key:           example.Key,
				Text:          example.Text,
				Embedding:     example.Embedding,
				ClusterSize:   1,
				AvgSimilarity: 1,
			})
		}
		return &prototypeBank{config: resolved, prototypes: prototypes}
	}

	similarityMatrix := buildSimilarityMatrix(deduped)
	clusters := clusterPrototypeExamples(deduped, similarityMatrix, float64(resolved.ClusterSimilarityThreshold))
	prototypes := make([]prototypeRepresentative, 0, len(clusters))
	for _, cluster := range clusters {
		prototypes = append(prototypes, selectPrototypeMedoid(deduped, cluster, similarityMatrix))
	}

	sort.Slice(prototypes, func(i, j int) bool {
		if prototypes[i].ClusterSize != prototypes[j].ClusterSize {
			return prototypes[i].ClusterSize > prototypes[j].ClusterSize
		}
		if prototypes[i].AvgSimilarity != prototypes[j].AvgSimilarity {
			return prototypes[i].AvgSimilarity > prototypes[j].AvgSimilarity
		}
		return prototypes[i].Text < prototypes[j].Text
	})

	if resolved.MaxPrototypes > 0 && len(prototypes) > resolved.MaxPrototypes {
		prototypes = prototypes[:resolved.MaxPrototypes]
	}

	return &prototypeBank{config: resolved, prototypes: prototypes}
}

func dedupePrototypeExamples(examples []prototypeExample) []prototypeExample {
	seen := make(map[string]struct{}, len(examples))
	deduped := make([]prototypeExample, 0, len(examples))
	for _, example := range examples {
		if len(example.Embedding) == 0 {
			continue
		}
		key := example.Key
		if key == "" {
			key = example.Text
		}
		if _, ok := seen[key]; ok {
			continue
		}
		seen[key] = struct{}{}
		deduped = append(deduped, example)
	}
	return deduped
}

func buildSimilarityMatrix(examples []prototypeExample) [][]float64 {
	matrix := make([][]float64, len(examples))
	for i := range examples {
		matrix[i] = make([]float64, len(examples))
		matrix[i][i] = 1
	}
	for i := 0; i < len(examples); i++ {
		for j := i + 1; j < len(examples); j++ {
			similarity := float64(cosineSimilarity(examples[i].Embedding, examples[j].Embedding))
			matrix[i][j] = similarity
			matrix[j][i] = similarity
		}
	}
	return matrix
}

func clusterPrototypeExamples(examples []prototypeExample, similarityMatrix [][]float64, threshold float64) [][]int {
	remaining := remainingPrototypeIndices(len(examples))

	clusters := make([][]int, 0, len(examples))
	for len(remaining) > 0 {
		bestCluster := selectBestPrototypeCluster(examples, similarityMatrix, threshold, remaining)
		clusters = append(clusters, bestCluster)
		removeClusterMembers(remaining, bestCluster)
	}

	return clusters
}

func remainingPrototypeIndices(count int) map[int]struct{} {
	remaining := make(map[int]struct{}, count)
	for i := 0; i < count; i++ {
		remaining[i] = struct{}{}
	}
	return remaining
}

func selectBestPrototypeCluster(
	examples []prototypeExample,
	similarityMatrix [][]float64,
	threshold float64,
	remaining map[int]struct{},
) []int {
	bestSeed := -1
	bestCluster := []int(nil)
	for idx := range remaining {
		cluster := clusterForPrototypeSeed(similarityMatrix, threshold, remaining, idx)
		if preferPrototypeCluster(examples, idx, bestSeed, cluster, bestCluster) {
			bestSeed = idx
			bestCluster = cluster
		}
	}
	return bestCluster
}

func clusterForPrototypeSeed(
	similarityMatrix [][]float64,
	threshold float64,
	remaining map[int]struct{},
	seed int,
) []int {
	cluster := make([]int, 0, len(remaining))
	cluster = append(cluster, seed)
	for other := range remaining {
		if other == seed {
			continue
		}
		if similarityMatrix[seed][other] >= threshold {
			cluster = append(cluster, other)
		}
	}
	sort.Ints(cluster)
	return cluster
}

func preferPrototypeCluster(
	examples []prototypeExample,
	candidateSeed int,
	currentSeed int,
	candidateCluster []int,
	currentCluster []int,
) bool {
	if len(candidateCluster) != len(currentCluster) {
		return len(candidateCluster) > len(currentCluster)
	}
	if currentSeed == -1 {
		return true
	}
	return examples[candidateSeed].Text < examples[currentSeed].Text
}

func removeClusterMembers(remaining map[int]struct{}, cluster []int) {
	for _, idx := range cluster {
		delete(remaining, idx)
	}
}

func selectPrototypeMedoid(examples []prototypeExample, cluster []int, similarityMatrix [][]float64) prototypeRepresentative {
	bestIndex := cluster[0]
	bestAvgSimilarity := -1.0

	for _, idx := range cluster {
		sum := 0.0
		for _, other := range cluster {
			sum += similarityMatrix[idx][other]
		}
		avg := sum / float64(len(cluster))
		if avg > bestAvgSimilarity || (avg == bestAvgSimilarity && examples[idx].Text < examples[bestIndex].Text) {
			bestIndex = idx
			bestAvgSimilarity = avg
		}
	}

	return prototypeRepresentative{
		Key:           examples[bestIndex].Key,
		Text:          examples[bestIndex].Text,
		Embedding:     examples[bestIndex].Embedding,
		ClusterSize:   len(cluster),
		AvgSimilarity: bestAvgSimilarity,
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

func (b *prototypeBank) representatives() []prototypeRepresentative {
	if b == nil || len(b.prototypes) == 0 {
		return nil
	}
	representatives := make([]prototypeRepresentative, len(b.prototypes))
	copy(representatives, b.prototypes)
	return representatives
}
