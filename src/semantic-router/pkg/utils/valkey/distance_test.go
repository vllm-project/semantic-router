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

package valkey

import (
	"math"
	"testing"
)

func TestDistanceToSimilarity(t *testing.T) {
	tests := []struct {
		name     string
		metric   string
		distance float64
		expected float64
	}{
		{"COSINE identical", "COSINE", 0.0, 1.0},
		{"COSINE orthogonal", "COSINE", 1.0, 0.5},
		{"COSINE opposite", "COSINE", 2.0, 0.0},
		// IP distance is 1 - u*v, so identical normalized vectors are at
		// distance 0 and must score 1.0, not 0.0.
		{"IP identical", "IP", 0.0, 1.0},
		{"IP orthogonal", "IP", 1.0, 0.0},
		{"IP inverts distance", "IP", 0.75, 0.25},
		{"L2 identical", "L2", 0.0, 1.0},
		{"L2 squashes", "L2", 1.0, 0.5},
		// Metric types are normalized, so a config using lowercase resolves to
		// the same branch as the canonical spelling.
		{"lowercase cosine", "cosine", 1.0, 0.5},
		{"lowercase ip", "ip", 0.75, 0.25},
		{"lowercase l2", "l2", 1.0, 0.5},
		{"unknown metric falls back", "EUCLIDEAN", 0.25, 0.75},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := DistanceToSimilarity(tt.metric, tt.distance)
			if math.Abs(got-tt.expected) > 0.001 {
				t.Errorf("DistanceToSimilarity(%q, %v) = %v, want %v",
					tt.metric, tt.distance, got, tt.expected)
			}
		})
	}
}

// TestDistanceToSimilarityIsDecreasing guards the property every caller depends
// on: they keep a result when similarity >= threshold, so a larger distance must
// never yield a larger similarity. A branch that returns the distance unchanged
// inverts that criterion and fails here.
func TestDistanceToSimilarityIsDecreasing(t *testing.T) {
	distances := []float64{0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0}

	for _, metric := range []string{"COSINE", "IP", "L2", "unknown"} {
		t.Run(metric, func(t *testing.T) {
			for i := 1; i < len(distances); i++ {
				closer := DistanceToSimilarity(metric, distances[i-1])
				farther := DistanceToSimilarity(metric, distances[i])
				if closer <= farther {
					t.Errorf("distance %v scored %v, not more similar than distance %v at %v",
						distances[i-1], closer, distances[i], farther)
				}
			}
			// An exact match must be maximally similar, so that repeating a
			// stored query clears any threshold in [0, 1].
			if got := DistanceToSimilarity(metric, 0.0); math.Abs(got-1.0) > 0.001 {
				t.Errorf("distance 0 scored %v, want 1.0", got)
			}
		})
	}
}
