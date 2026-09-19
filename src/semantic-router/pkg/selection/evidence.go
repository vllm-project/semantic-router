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
	"cmp"
	"fmt"

	modelcatalog "github.com/vllm-project/semantic-router/src/semantic-router/pkg/catalog"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

type candidateRank struct {
	score    float64
	evidence *modelcatalog.IndexResult
}

func (rank candidateRank) Compare(other candidateRank) int {
	if order := cmp.Compare(rank.score, other.score); order != 0 {
		return order
	}
	if !comparableEvidence(rank.evidence, other.evidence) || *rank.evidence.Score != *other.evidence.Score {
		return 0
	}
	return cmp.Compare(rank.evidence.Coverage, other.evidence.Coverage)
}

func comparableEvidence(left, right *modelcatalog.IndexResult) bool {
	return left != nil && right != nil &&
		left.Status == "available" && right.Status == "available" &&
		left.Score != nil && right.Score != nil && left.Index == right.Index
}

func evidenceForCandidate(modelParams map[string]config.ModelParams, candidate config.ModelRef) *modelcatalog.IndexResult {
	params, ok := modelParams[candidate.Model]
	if !ok {
		return nil
	}
	result, ok := params.EvidenceResultAt("", candidate.ReasoningEffort)
	if !ok || result.Status != "available" || result.Score == nil {
		return nil
	}
	return &result
}

func candidateScoreKey(candidates []config.ModelRef, index int) string {
	model := candidates[index].Model
	for i := range candidates {
		if i != index && candidates[i].Model == model {
			return fmt.Sprintf("%s[%d,effort=%s]", model, index, candidates[index].ReasoningEffort)
		}
	}
	return model
}

func evidenceDiagnostic(evidence *modelcatalog.IndexResult) string {
	if evidence == nil || evidence.Score == nil || evidence.Status != "available" {
		return "intelligence{status=unavailable}"
	}
	return fmt.Sprintf("intelligence{index=%s effort=%s score=%.2f coverage=%.0f%%}",
		evidence.Index, evidence.ReasoningEffort, *evidence.Score, evidence.Coverage*100)
}
