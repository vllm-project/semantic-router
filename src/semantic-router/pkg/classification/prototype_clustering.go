package classification

import "sort"

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

func buildPrototypeSimilarityMatrix(examples []prototypeExample) [][]float64 {
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
