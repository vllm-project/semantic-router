package extproc

import (
	"math"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/latency"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
)

func currentLearningModel(selCtx *selection.SelectionContext) string {
	if selCtx == nil || selCtx.AgenticSession == nil {
		return ""
	}
	return strings.TrimSpace(selCtx.AgenticSession.PreviousModel)
}

func selectionContextContainsModel(selCtx *selection.SelectionContext, model string) bool {
	if selCtx == nil || model == "" {
		return false
	}
	for _, candidate := range selCtx.CandidateModels {
		if candidate.Model == model || candidate.LoRAName == model {
			return true
		}
	}
	return false
}

func (r *OpenAIRouter) configuredBackendModel(model string) bool {
	if r == nil || r.Config == nil || strings.TrimSpace(model) == "" {
		return false
	}
	if _, ok := r.Config.ModelConfig[model]; ok {
		return true
	}
	return model == r.Config.DefaultModel
}

func (r *OpenAIRouter) eligibleLearningModelRefs(refs []config.ModelRef) []config.ModelRef {
	if len(refs) == 0 {
		return nil
	}
	eligible := make([]config.ModelRef, 0, len(refs))
	for _, ref := range refs {
		if strings.TrimSpace(ref.Model) == "" || !r.configuredBackendModel(ref.Model) {
			continue
		}
		eligible = append(eligible, ref)
	}
	return eligible
}

func cloneSelectionScores(scores map[string]float64) map[string]float64 {
	if scores == nil {
		return nil
	}
	clone := make(map[string]float64, len(scores))
	for key, value := range scores {
		clone[key] = value
	}
	return clone
}

// estimateGateCacheWarmth produces a request-time cache warmth prior for model.
// It estimates ambient cache warmth from TTFT update freshness, which is the
// only cache signal available before the current request executes.
func estimateGateCacheWarmth(model string, now time.Time) (float64, bool) {
	if model == "" {
		return 0, false
	}
	lastUpdated, ok := latency.GetTTFTLastUpdated(model)
	if !ok {
		return 0, false
	}
	if now.IsZero() {
		now = time.Now()
	}
	age := now.Sub(lastUpdated).Seconds()
	if age < 0 {
		age = 0
	}
	warmth := math.Exp(-math.Ln2 * age / latency.FreshnessHalfLifeSeconds)
	if warmth < 0 {
		warmth = 0
	}
	if warmth > 1 {
		warmth = 1
	}
	return warmth, true
}
