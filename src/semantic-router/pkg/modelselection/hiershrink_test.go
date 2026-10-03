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
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

var (
	hierLeft  = []float64{-1, 0}
	hierRight = []float64{1, 0}
)

func hierFeature(embedding []float64) []float64 {
	return CombineEmbeddingWithCategory(embedding, "math")
}

func hierRecords(model string, embedding []float64, quality float64, latencyNs int64, n int) []TrainingRecord {
	records := make([]TrainingRecord, n)
	for i := range records {
		records[i] = TrainingRecord{
			QueryEmbedding:    hierFeature(embedding),
			SelectedModel:     model,
			ResponseQuality:   quality,
			ResponseLatencyNs: latencyNs,
		}
	}
	return records
}

func newHierShrinkTestSelector(t *testing.T, costWeight float64, training []TrainingRecord) *HierShrinkSelector {
	t.Helper()
	centroids := [][]float64{hierFeature(hierLeft), hierFeature(hierRight)}
	data, err := json.Marshal(hierShrinkArtifact{
		Algorithm:       "hiershrink",
		CoarseCentroids: centroids,
		FineCentroids:   centroids,
		CostWeight:      costWeight,
		Training:        training,
	})
	if err != nil {
		t.Fatal(err)
	}
	selector := NewHierShrinkSelector()
	if err := selector.LoadFromJSON(data); err != nil {
		t.Fatalf("LoadFromJSON: %v", err)
	}
	return selector
}

func selectHierShrink(t *testing.T, selector *HierShrinkSelector, embedding []float64, refs []config.ModelRef) string {
	t.Helper()
	ref, err := selector.Select(&SelectionContext{QueryEmbedding: embedding, CategoryName: "math"}, refs)
	if err != nil {
		t.Fatalf("Select: %v", err)
	}
	return ref.Model
}

func TestHierShrink_RoutesSparseCandidateByCluster(t *testing.T) {
	training := append(hierRecords("model-a", hierLeft, 0.6, 1, 500), hierRecords("model-a", hierRight, 0.6, 1, 500)...)
	training = append(training, hierRecords("model-b", hierLeft, 1.0, 1, 30)...)
	training = append(training, hierRecords("model-b", hierRight, 0.0, 1, 30)...)
	selector := newHierShrinkTestSelector(t, 0, training)

	if got := selectHierShrink(t, selector, hierLeft, testModels); got != "model-b" {
		t.Errorf("left query selected %s, want model-b", got)
	}
	if got := selectHierShrink(t, selector, hierRight, testModels); got != "model-a" {
		t.Errorf("right query selected %s, want model-a", got)
	}
}

func TestHierShrink_EmptyClusterFallsBackToGlobalMean(t *testing.T) {
	training := append(hierRecords("model-a", hierLeft, 0.9, 1, 10), hierRecords("model-b", hierRight, 0.5, 1, 100)...)
	selector := newHierShrinkTestSelector(t, 0, training)

	if got := selector.models["model-a"].estimate(1, 1); math.Abs(got-0.9) > 1e-12 {
		t.Errorf("estimate in empty cluster = %v, want global mean 0.9", got)
	}
	if got := selectHierShrink(t, selector, hierRight, testModels); got != "model-a" {
		t.Errorf("selected %s, want model-a", got)
	}
}

func TestHierShrink_EstimateShrinksTowardClusterMean(t *testing.T) {
	training := append(hierRecords("model-a", hierLeft, 1.0, 1, 24), hierRecords("model-a", hierRight, 0.0, 1, 24)...)
	selector := newHierShrinkTestSelector(t, 0, training)

	b1 := (24*1.0 + 24*0.5) / 48
	want := (24*1.0 + 24*b1) / 48
	if got := selector.models["model-a"].estimate(0, 0); math.Abs(got-want) > 1e-12 {
		t.Errorf("estimate = %v, want %v", got, want)
	}
}

func TestHierShrink_CostWeightPrefersCheaperModel(t *testing.T) {
	training := append(hierRecords("model-a", hierLeft, 0.8, 1000, 100), hierRecords("model-b", hierLeft, 0.7, 100, 100)...)

	if got := selectHierShrink(t, newHierShrinkTestSelector(t, 0, training), hierLeft, testModels); got != "model-a" {
		t.Errorf("cost_weight 0 selected %s, want model-a", got)
	}
	if got := selectHierShrink(t, newHierShrinkTestSelector(t, 0.5, training), hierLeft, testModels); got != "model-b" {
		t.Errorf("cost_weight 0.5 selected %s, want model-b", got)
	}
}

func TestHierShrink_SkipsCandidatesWithoutRecords(t *testing.T) {
	selector := newHierShrinkTestSelector(t, 0, hierRecords("model-b", hierLeft, 0.1, 1, 5))
	if got := selectHierShrink(t, selector, hierLeft, testModels); got != "model-b" {
		t.Errorf("selected %s, want model-b", got)
	}

	unknown := []config.ModelRef{{Model: "model-x"}, {Model: "model-y"}}
	if _, err := selector.Select(&SelectionContext{QueryEmbedding: hierLeft}, unknown); err == nil {
		t.Error("expected error when no candidate has records")
	}
}

func TestHierShrink_Errors(t *testing.T) {
	if _, err := NewHierShrinkSelector().Select(&SelectionContext{QueryEmbedding: hierLeft}, testModels); err == nil {
		t.Error("expected error for untrained selector")
	}
	if err := NewHierShrinkSelector().Train(hierRecords("model-a", hierLeft, 1, 1, 1)); err == nil {
		t.Error("expected error when training before centroids load")
	}

	selector := newHierShrinkTestSelector(t, 0, hierRecords("model-a", hierLeft, 1, 1, 1))
	if _, err := selector.Select(&SelectionContext{QueryEmbedding: []float64{1, 0, 0}}, testModels); err == nil {
		t.Error("expected error for feature dimension mismatch")
	}
	if err := selector.Train([]TrainingRecord{{QueryEmbedding: []float64{1}, SelectedModel: "model-a"}}); err == nil {
		t.Error("expected error for record dimension mismatch")
	}

	for name, body := range map[string]string{
		"wrong algorithm":   `{"algorithm":"knn","coarse_centroids":[[1]],"fine_centroids":[[1]]}`,
		"missing centroids": `{"algorithm":"hiershrink"}`,
		"ragged centroids":  `{"algorithm":"hiershrink","coarse_centroids":[[1]],"fine_centroids":[[1,2]]}`,
		"negative cost":     `{"algorithm":"hiershrink","coarse_centroids":[[1]],"fine_centroids":[[1]],"cost_weight":-1}`,
	} {
		if err := NewHierShrinkSelector().LoadFromJSON([]byte(body)); err == nil {
			t.Errorf("%s: expected load error", name)
		}
	}
}

func TestHierShrink_LoadPretrainedFromPath(t *testing.T) {
	centroids := [][]float64{hierFeature(hierLeft), hierFeature(hierRight)}
	data, err := json.Marshal(hierShrinkArtifact{
		Algorithm:       "hiershrink",
		CoarseCentroids: centroids,
		FineCentroids:   centroids,
		Training:        hierRecords("model-b", hierRight, 0.9, 1, 3),
	})
	if err != nil {
		t.Fatal(err)
	}
	dir := t.TempDir()
	if err = os.WriteFile(filepath.Join(dir, "hiershrink_model.json"), data, 0o644); err != nil {
		t.Fatal(err)
	}

	selector, err := NewSelector(&config.MLModelSelectionConfig{Type: "hiershrink", ModelsPath: dir})
	if err != nil {
		t.Fatalf("NewSelector: %v", err)
	}
	ref, err := selector.Select(&SelectionContext{QueryEmbedding: hierRight, CategoryName: "math"}, testModels)
	if err != nil {
		t.Fatalf("Select: %v", err)
	}
	if ref.Model != "model-b" {
		t.Errorf("selected %s, want model-b", ref.Model)
	}
}
