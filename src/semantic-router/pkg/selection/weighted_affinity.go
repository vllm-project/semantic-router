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

package selection

import (
	"crypto/sha256"
	"encoding/binary"
	"math"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// WeightedModelRefForAffinity chooses one ModelRef from an explicitly weighted
// tier. The opaque affinity scope must contain stable request and routing
// identity; the hash makes the result identical across Router replicas without
// shared mutable state.
func WeightedModelRefForAffinity(
	candidates []config.ModelRef,
	affinityScope string,
) (*config.ModelRef, map[string]float64, bool) {
	if len(candidates) == 0 {
		return nil, nil, false
	}
	total := 0.0
	for _, candidate := range candidates {
		if candidate.Weight <= 0 || math.IsNaN(candidate.Weight) || math.IsInf(candidate.Weight, 0) {
			return nil, nil, false
		}
		total += candidate.Weight
		if math.IsInf(total, 0) {
			return nil, nil, false
		}
	}
	if total <= 0 {
		return nil, nil, false
	}
	if len(candidates) == 1 {
		return weightedModelRefAt(candidates, total, 0)
	}
	if affinityScope == "" {
		return nil, nil, false
	}

	digest := sha256.Sum256([]byte(affinityScope))
	// A 53-bit fraction is represented exactly by float64. Keeping the sample
	// strictly below one makes the final cumulative bucket exhaustive.
	sample := float64(binary.BigEndian.Uint64(digest[:8])>>11) / float64(uint64(1)<<53)
	return weightedModelRefAt(candidates, total, sample)
}

func weightedModelRefAt(
	candidates []config.ModelRef,
	total float64,
	sample float64,
) (*config.ModelRef, map[string]float64, bool) {
	if len(candidates) == 0 || total <= 0 || sample < 0 || sample >= 1 {
		return nil, nil, false
	}
	threshold := sample * total
	cumulative := 0.0
	scores := make(map[string]float64, len(candidates))
	for _, candidate := range candidates {
		scores[candidate.Model] = candidate.Weight / total
	}
	for index := range candidates {
		candidate := &candidates[index]
		cumulative += candidate.Weight
		if threshold < cumulative {
			return candidate, scores, true
		}
	}
	return &candidates[len(candidates)-1], scores, true
}
