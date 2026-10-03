package modelselection

import (
	"encoding/json"
	"errors"
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding/vecmath"
)

// kmeansModel is a KMeans artifact: cluster centroids and the model
// assigned to each cluster.
type kmeansModel struct {
	dim       int
	centroids []float64 // clusters × dim, row-major
	models    []string
}

func parseKMeans(data []byte) (classifier, error) {
	var artifact struct {
		Algorithm     string      `json:"algorithm"`
		Trained       bool        `json:"trained"`
		Centroids     [][]float64 `json:"centroids"`
		ClusterModels []string    `json:"cluster_models"`
	}
	if err := json.Unmarshal(data, &artifact); err != nil {
		return nil, err
	}
	if artifact.Algorithm != "kmeans" {
		return nil, errors.New("unsupported KMeans artifact format")
	}
	if !artifact.Trained {
		return nil, errors.New("artifact is not trained")
	}
	if len(artifact.Centroids) == 0 || len(artifact.Centroids[0]) == 0 || len(artifact.ClusterModels) != len(artifact.Centroids) {
		return nil, errors.New("invalid KMeans centroids or cluster models")
	}
	model := &kmeansModel{dim: len(artifact.Centroids[0]), models: artifact.ClusterModels}
	for _, centroid := range artifact.Centroids {
		if len(centroid) != model.dim || !finite(centroid) {
			return nil, errors.New("invalid KMeans centroids or cluster models")
		}
		model.centroids = append(model.centroids, centroid...)
	}
	return model, nil
}

// classify assigns the query to the nearest centroid; ties go to the lower
// cluster index.
func (m *kmeansModel) classify(query []float64) (string, error) {
	if len(query) != m.dim || !finite(query) {
		return "", fmt.Errorf("expected %d finite KMeans features, got %d", m.dim, len(query))
	}
	nearest, best := 0, 0.0
	for i := range m.models {
		distance := vecmath.SquaredDistance64(m.centroids[i*m.dim:(i+1)*m.dim], query)
		if i == 0 || distance < best {
			nearest, best = i, distance
		}
	}
	return m.models[nearest], nil
}
