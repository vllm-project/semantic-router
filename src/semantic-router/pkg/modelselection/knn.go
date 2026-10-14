package modelselection

import (
	"container/heap"
	"encoding/json"
	"errors"
	"fmt"
	"sort"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding/vecmath"
)

// knnModel is a KNN artifact: unit-normalized training features, their
// labels, and each sample's vote weight (0.9·quality + 0.1·speed).
type knnModel struct {
	k        int
	dim      int
	features []float64 // samples × dim, row-major
	labels   []int     // index into names
	names    []string  // distinct labels in lexicographic order
	weights  []float64
}

func parseKNN(data []byte) (classifier, error) {
	var artifact struct {
		Algorithm     string      `json:"algorithm"`
		FormatVersion *int        `json:"format_version"`
		Trained       bool        `json:"trained"`
		K             int         `json:"k"`
		Embeddings    [][]float64 `json:"embeddings"`
		Labels        []string    `json:"labels"`
		Qualities     []float64   `json:"qualities"`
		Latencies     []int64     `json:"latencies"`
	}
	if err := json.Unmarshal(data, &artifact); err != nil {
		return nil, err
	}
	if artifact.Algorithm != "knn" || (artifact.FormatVersion != nil && *artifact.FormatVersion != 1 && *artifact.FormatVersion != 2) {
		return nil, errors.New("unsupported KNN artifact format")
	}
	n := len(artifact.Embeddings)
	if len(artifact.Qualities) == 0 {
		artifact.Qualities = make([]float64, n)
		for i := range artifact.Qualities {
			artifact.Qualities[i] = 0.5
		}
	}
	if len(artifact.Latencies) == 0 {
		artifact.Latencies = make([]int64, n)
	}
	if !artifact.Trained {
		return nil, errors.New("artifact is not trained")
	}
	if artifact.K <= 0 || n == 0 || len(artifact.Embeddings[0]) == 0 || len(artifact.Labels) != n ||
		len(artifact.Qualities) != n || len(artifact.Latencies) != n || !finite(artifact.Qualities) {
		return nil, errors.New("invalid KNN sample shapes or values")
	}
	model := &knnModel{k: min(artifact.K, n), dim: len(artifact.Embeddings[0])}
	model.features = make([]float64, 0, n*model.dim)
	model.weights = make([]float64, n)
	for i, row := range artifact.Embeddings {
		if len(row) != model.dim || !finite(row) || artifact.Latencies[i] < 0 || artifact.Labels[i] == "" {
			return nil, errors.New("invalid KNN sample shapes or values")
		}
		model.features = append(model.features, normalized(row)...)
		speed := 1 / (1 + float64(artifact.Latencies[i])/10_000_000_000)
		model.weights[i] = float64(0.9*artifact.Qualities[i]) + float64(0.1*speed)
	}
	model.names, model.labels = indexLabels(artifact.Labels)
	return model, nil
}

// normalized divides v by its Euclidean norm, leaving a zero vector as is.
func normalized(v []float64) []float64 {
	out := make([]float64, len(v))
	n := norm(v)
	for i, x := range v {
		if n > 0 {
			out[i] = x / n
		} else {
			out[i] = x
		}
	}
	return out
}

// indexLabels returns the distinct labels in lexicographic order and each
// sample's index into them.
func indexLabels(labels []string) ([]string, []int) {
	seen := make(map[string]bool)
	var names []string
	for _, label := range labels {
		if !seen[label] {
			seen[label] = true
			names = append(names, label)
		}
	}
	sort.Strings(names)
	index := make(map[string]int, len(names))
	for i, name := range names {
		index[name] = i
	}
	ids := make([]int, len(labels))
	for i, label := range labels {
		ids[i] = index[label]
	}
	return names, ids
}

func (m *knnModel) classify(query []float64) (string, error) {
	if len(query) != m.dim || !finite(query) {
		return "", fmt.Errorf("expected %d finite KNN features, got %d", m.dim, len(query))
	}
	query = normalized(query)
	// The k nearest samples by squared distance, ties to the lower index.
	nearest := make(neighborHeap, 0, m.k)
	for i := 0; i*m.dim < len(m.features); i++ {
		candidate := neighbor{distance: vecmath.SquaredDistance64(m.features[i*m.dim:(i+1)*m.dim], query), index: i}
		if len(nearest) < m.k {
			heap.Push(&nearest, candidate)
		} else if candidate.closer(nearest[0]) {
			nearest[0] = candidate
			heap.Fix(&nearest, 0)
		}
	}
	// Votes accumulate nearest first; among the voted labels the first in
	// lexicographic order wins a tie.
	sort.Slice(nearest, func(a, b int) bool { return nearest[a].closer(nearest[b]) })
	votes := make([]float64, len(m.names))
	voted := make([]bool, len(m.names))
	for _, n := range nearest {
		votes[m.labels[n.index]] += m.weights[n.index]
		voted[m.labels[n.index]] = true
	}
	winner := -1
	for i := range votes {
		if voted[i] && (winner < 0 || votes[i] > votes[winner]) {
			winner = i
		}
	}
	return m.names[winner], nil
}

type neighbor struct {
	distance float64
	index    int
}

func (n neighbor) closer(other neighbor) bool {
	return n.distance < other.distance || (n.distance == other.distance && n.index < other.index)
}

// neighborHeap is a max-heap: the farthest kept neighbor is at the root.
type neighborHeap []neighbor

func (h neighborHeap) Len() int            { return len(h) }
func (h neighborHeap) Less(i, j int) bool  { return h[j].closer(h[i]) }
func (h neighborHeap) Swap(i, j int)       { h[i], h[j] = h[j], h[i] }
func (h *neighborHeap) Push(x interface{}) { *h = append(*h, x.(neighbor)) }
func (h *neighborHeap) Pop() interface{} {
	old := *h
	last := old[len(old)-1]
	*h = old[:len(old)-1]
	return last
}
