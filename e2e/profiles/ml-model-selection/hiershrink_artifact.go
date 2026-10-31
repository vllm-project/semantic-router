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

package mlmodelselection

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
)

func writeHierShrinkArtifact(dir string) error {
	var knn struct {
		Embeddings [][]float64 `json:"embeddings"`
		Labels     []string    `json:"labels"`
		Qualities  []float64   `json:"qualities"`
		Latencies  []int64     `json:"latencies"`
	}
	var kmeans struct {
		Centroids [][]float64 `json:"centroids"`
	}
	if err := readJSON(filepath.Join(dir, "knn_model.json"), &knn); err != nil {
		return err
	}
	if err := readJSON(filepath.Join(dir, "kmeans_model.json"), &kmeans); err != nil {
		return err
	}
	if len(knn.Labels) != len(knn.Embeddings) || len(knn.Labels) != len(knn.Qualities) || len(knn.Labels) != len(knn.Latencies) {
		return fmt.Errorf("knn_model.json sample arrays differ in length")
	}

	training := make([]map[string]interface{}, len(knn.Labels))
	for i, label := range knn.Labels {
		training[i] = map[string]interface{}{
			"query_embedding":     knn.Embeddings[i],
			"selected_model":      label,
			"response_quality":    knn.Qualities[i],
			"response_latency_ns": knn.Latencies[i],
			"success":             true,
		}
	}
	data, err := json.Marshal(map[string]interface{}{
		"algorithm":        "hiershrink",
		"coarse_centroids": kmeans.Centroids,
		"fine_centroids":   kmeans.Centroids,
		"training":         training,
	})
	if err != nil {
		return err
	}
	return os.WriteFile(filepath.Join(dir, "hiershrink_model.json"), data, 0o644)
}

func readJSON(path string, v interface{}) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	return json.Unmarshal(data, v)
}
