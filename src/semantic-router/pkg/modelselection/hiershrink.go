/*
Copyright 2025 vLLM Semantic Router.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package modelselection

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"sync"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

const hierShrinkLambda = 24.0

type HierShrinkSelector struct {
	mu         sync.RWMutex
	coarse     [][]float64
	fine       [][]float64
	costWeight float64
	models     map[string]*hierShrinkStats
}

type hierShrinkStats struct {
	sum         float64
	count       float64
	latencySum  float64
	coarseSum   []float64
	coarseCount []float64
	fineSum     []float64
	fineCount   []float64
}

type hierShrinkArtifact struct {
	Algorithm       string           `json:"algorithm"`
	CoarseCentroids [][]float64      `json:"coarse_centroids"`
	FineCentroids   [][]float64      `json:"fine_centroids"`
	CostWeight      float64          `json:"cost_weight"`
	Training        []TrainingRecord `json:"training"`
}

func NewHierShrinkSelector() *HierShrinkSelector {
	return &HierShrinkSelector{models: make(map[string]*hierShrinkStats)}
}

func (s *HierShrinkSelector) Name() string { return "hiershrink" }

func (s *HierShrinkSelector) Load(path string) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return fmt.Errorf("failed to read HierShrink model file: %w", err)
	}
	return s.LoadFromJSON(data)
}

func (s *HierShrinkSelector) LoadFromJSON(data []byte) error {
	var artifact hierShrinkArtifact
	if err := json.Unmarshal(data, &artifact); err != nil {
		return fmt.Errorf("failed to parse HierShrink model JSON: %w", err)
	}
	if artifact.Algorithm != "hiershrink" {
		return fmt.Errorf("HierShrink: unexpected algorithm %q", artifact.Algorithm)
	}
	if err := validateCentroids(artifact.CoarseCentroids, artifact.FineCentroids); err != nil {
		return err
	}
	if math.IsNaN(artifact.CostWeight) || math.IsInf(artifact.CostWeight, 0) || artifact.CostWeight < 0 {
		return fmt.Errorf("HierShrink: cost_weight must be finite and nonnegative")
	}

	s.mu.Lock()
	s.coarse = artifact.CoarseCentroids
	s.fine = artifact.FineCentroids
	s.costWeight = artifact.CostWeight
	s.models = make(map[string]*hierShrinkStats)
	s.mu.Unlock()

	return s.Train(artifact.Training)
}

func validateCentroids(coarse, fine [][]float64) error {
	if len(coarse) == 0 || len(fine) == 0 {
		return fmt.Errorf("HierShrink: coarse and fine centroids are required")
	}
	dim := len(coarse[0])
	for _, level := range [][][]float64{coarse, fine} {
		for _, centroid := range level {
			if len(centroid) != dim || dim == 0 {
				return fmt.Errorf("HierShrink: centroids must share one nonzero dimension")
			}
		}
	}
	return nil
}

func (s *HierShrinkSelector) Train(data []TrainingRecord) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if len(s.fine) == 0 {
		return fmt.Errorf("HierShrink: load centroids before training")
	}
	dim := len(s.fine[0])
	for _, record := range data {
		if record.SelectedModel == "" || len(record.QueryEmbedding) != dim {
			return fmt.Errorf("HierShrink: record needs a model name and a %d-dim feature vector", dim)
		}
		if math.IsNaN(record.ResponseQuality) || math.IsInf(record.ResponseQuality, 0) || record.ResponseLatencyNs < 0 {
			return fmt.Errorf("HierShrink: record needs finite quality and nonnegative latency")
		}
	}

	for _, record := range data {
		stats, ok := s.models[record.SelectedModel]
		if !ok {
			stats = &hierShrinkStats{
				coarseSum:   make([]float64, len(s.coarse)),
				coarseCount: make([]float64, len(s.coarse)),
				fineSum:     make([]float64, len(s.fine)),
				fineCount:   make([]float64, len(s.fine)),
			}
			s.models[record.SelectedModel] = stats
		}
		c := nearestCentroid(s.coarse, record.QueryEmbedding)
		f := nearestCentroid(s.fine, record.QueryEmbedding)
		stats.sum += record.ResponseQuality
		stats.count++
		stats.latencySum += float64(record.ResponseLatencyNs)
		stats.coarseSum[c] += record.ResponseQuality
		stats.coarseCount[c]++
		stats.fineSum[f] += record.ResponseQuality
		stats.fineCount[f]++
	}
	return nil
}

func (st *hierShrinkStats) estimate(coarse, fine int) float64 {
	mean := st.sum / st.count
	b1 := (st.coarseSum[coarse] + hierShrinkLambda*mean) / (st.coarseCount[coarse] + hierShrinkLambda)
	return (st.fineSum[fine] + hierShrinkLambda*b1) / (st.fineCount[fine] + hierShrinkLambda)
}

func nearestCentroid(centroids [][]float64, x []float64) int {
	best := 0
	bestDist := math.Inf(1)
	for i, centroid := range centroids {
		var dist float64
		for j, v := range centroid {
			d := x[j] - v
			dist += d * d
		}
		if dist < bestDist {
			best, bestDist = i, dist
		}
	}
	return best
}

func (s *HierShrinkSelector) Select(ctx *SelectionContext, refs []config.ModelRef) (*config.ModelRef, error) {
	if len(refs) == 0 {
		return nil, fmt.Errorf("HierShrink: no model refs provided")
	}
	if len(refs) == 1 {
		return &refs[0], nil
	}

	s.mu.RLock()
	defer s.mu.RUnlock()

	if len(s.fine) == 0 {
		return nil, fmt.Errorf("HierShrink: model not trained - load pretrained model first")
	}
	if len(ctx.QueryEmbedding) == 0 {
		return nil, fmt.Errorf("HierShrink: no query embedding provided")
	}
	featureVector := CombineEmbeddingWithCategory(ctx.QueryEmbedding, ctx.CategoryName)
	if len(featureVector) != len(s.fine[0]) {
		return nil, fmt.Errorf("HierShrink: feature dimension %d does not match artifact dimension %d", len(featureVector), len(s.fine[0]))
	}
	coarse := nearestCentroid(s.coarse, featureVector)
	fine := nearestCentroid(s.fine, featureVector)

	maxLatency := 0.0
	for _, ref := range refs {
		if stats, ok := s.models[getModelName(ref)]; ok {
			maxLatency = math.Max(maxLatency, stats.latencySum/stats.count)
		}
	}

	best := -1
	bestScore := math.Inf(-1)
	for i, ref := range refs {
		stats, ok := s.models[getModelName(ref)]
		if !ok {
			continue
		}
		score := stats.estimate(coarse, fine)
		if maxLatency > 0 {
			score -= s.costWeight * stats.latencySum / stats.count / maxLatency
		}
		if score > bestScore {
			best, bestScore = i, score
		}
	}
	if best < 0 {
		return nil, fmt.Errorf("HierShrink: no candidate model has training records")
	}

	logging.Infof("HierShrink selected model %s", getModelName(refs[best]))
	return &refs[best], nil
}
