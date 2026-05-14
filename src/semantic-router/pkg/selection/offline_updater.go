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
	"context"
	"fmt"
	"sort"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// OfflineUpdaterConfig configures the offline weight update process.
type OfflineUpdaterConfig struct {
	// ImplicitFeedbackWeight controls weight of auto-detected feedback signals
	// during offline replay (same semantics as RLDrivenConfig.ImplicitFeedbackWeight).
	ImplicitFeedbackWeight float64

	// TransitionPenaltyWeight controls how much model-switch transitions
	// penalize the switched-away model's Beta distribution.
	TransitionPenaltyWeight float64

	// MinRecordsPerModel is the minimum number of observations required before
	// a model's offline weights are considered meaningful. Models below this
	// threshold keep their prior-version weights unchanged.
	MinRecordsPerModel int

	// ParentVersionID is the version these weights are derived from.
	// The offline updater blends new evidence with the parent's priors.
	ParentVersionID string
}

// DefaultOfflineUpdaterConfig returns sensible defaults.
func DefaultOfflineUpdaterConfig() *OfflineUpdaterConfig {
	return &OfflineUpdaterConfig{
		ImplicitFeedbackWeight:  0.5,
		TransitionPenaltyWeight: 0.3,
		MinRecordsPerModel:      5,
	}
}

// OfflineUpdater replays an OfflineDataset and produces a new PolicyVersion
// with updated Beta distribution weights. It does not touch runtime state;
// the caller is responsible for persisting and activating the result.
type OfflineUpdater struct {
	config *OfflineUpdaterConfig
}

// NewOfflineUpdater creates a new offline updater.
func NewOfflineUpdater(cfg *OfflineUpdaterConfig) *OfflineUpdater {
	if cfg == nil {
		cfg = DefaultOfflineUpdaterConfig()
	}
	return &OfflineUpdater{config: cfg}
}

// Update replays the dataset and returns a candidate PolicyVersion.
// If parentWeights is non-nil, the new weights are blended with the parent's
// priors so that models with few observations retain stable estimates.
func (u *OfflineUpdater) Update(ctx context.Context, dataset *OfflineDataset, parentWeights *PolicyWeights) (*PolicyVersion, error) {
	if dataset == nil || len(dataset.Records) == 0 {
		return nil, fmt.Errorf("empty dataset")
	}

	// Accumulate per-scope Beta distribution updates.
	global := make(map[string]*betaAccumulator)
	category := make(map[string]map[string]*betaAccumulator)
	user := make(map[string]map[string]*betaAccumulator)

	for _, rec := range dataset.Records {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
		}

		weight := u.recordWeight(rec)

		// Apply outcome feedback to the selected model.
		u.applyOutcome(global, rec.SelectedModel, rec.Outcome, weight)

		if rec.DecisionName != "" {
			if category[rec.DecisionName] == nil {
				category[rec.DecisionName] = make(map[string]*betaAccumulator)
			}
			u.applyOutcome(category[rec.DecisionName], rec.SelectedModel, rec.Outcome, weight)
		}

		if rec.UserID != "" {
			if user[rec.UserID] == nil {
				user[rec.UserID] = make(map[string]*betaAccumulator)
			}
			u.applyOutcome(user[rec.UserID], rec.SelectedModel, rec.Outcome, weight)
		}

		// Apply transition penalty if a model switch occurred.
		if rec.TransitionEvidence != nil && rec.TransitionEvidence.PreviousModel != "" {
			penalty := u.config.TransitionPenaltyWeight
			acc := u.getOrCreate(global, rec.TransitionEvidence.PreviousModel)
			acc.beta += penalty
			acc.count++
		}
	}

	// Build PolicyWeights from accumulators, blending with parent priors.
	weights := PolicyWeights{
		Global:   u.finalize(global, parentGlobal(parentWeights)),
		Category: u.finalizeScoped(category, parentCategory(parentWeights)),
		User:     u.finalizeScoped(user, parentUser(parentWeights)),
	}

	versionID := fmt.Sprintf("offline-%s", time.Now().UTC().Format("20060102-150405"))
	version := &PolicyVersion{
		ID:        versionID,
		ParentID:  u.config.ParentVersionID,
		CreatedAt: time.Now().UTC(),
		Source:    "offline_batch",
		DatasetWindow: &TimeWindow{
			Start: dataset.WindowStart,
			End:   dataset.WindowEnd,
		},
		TrainingStats: &TrainingStats{
			RecordCount:     len(dataset.Records),
			SessionCount:    dataset.SessionCount,
			TransitionCount: dataset.TransitionCount,
			FeedbackCount:   countFeedback(dataset),
		},
		Status:  PolicyStatusCandidate,
		Weights: weights,
	}

	logging.ComponentEvent("selection", "offline_update_complete", map[string]interface{}{
		"version_id":         versionID,
		"record_count":       len(dataset.Records),
		"session_count":      dataset.SessionCount,
		"global_model_count": len(weights.Global),
	})

	return version, nil
}

// betaAccumulator tracks incremental Beta distribution updates.
type betaAccumulator struct {
	alpha float64
	beta  float64
	count int
}

func (u *OfflineUpdater) getOrCreate(m map[string]*betaAccumulator, model string) *betaAccumulator {
	acc, ok := m[model]
	if !ok {
		acc = &betaAccumulator{}
		m[model] = acc
	}
	return acc
}

func (u *OfflineUpdater) recordWeight(rec OfflineDatasetRecord) float64 {
	weight := 1.0
	if strings.HasPrefix(rec.Outcome.FeedbackType, "implicit") {
		weight = u.config.ImplicitFeedbackWeight
		if rec.Outcome.FeedbackConfidence > 0 {
			weight *= rec.Outcome.FeedbackConfidence
		}
	}
	return weight
}

func (u *OfflineUpdater) applyOutcome(scope map[string]*betaAccumulator, model string, outcome OfflineOutcome, weight float64) {
	acc := u.getOrCreate(scope, model)
	acc.count++

	if outcome.Won != nil {
		applyWinLoss(acc, *outcome.Won, outcome.Tie, weight)
	}

	// Apply opponent's inverse outcome.
	if outcome.OpponentModel != "" && outcome.Won != nil {
		opp := u.getOrCreate(scope, outcome.OpponentModel)
		opp.count++
		applyWinLoss(opp, !*outcome.Won, outcome.Tie, weight)
	}

	// Implicit outcome from response status.
	if outcome.ResponseStatus >= 500 {
		acc.beta += weight * 0.5
	}
}

func applyWinLoss(acc *betaAccumulator, won, tie bool, weight float64) {
	if tie {
		acc.alpha += 0.5 * weight
		acc.beta += 0.5 * weight
		return
	}
	if won {
		acc.alpha += weight
		return
	}
	acc.beta += weight
}

func (u *OfflineUpdater) finalize(accs map[string]*betaAccumulator, parent map[string]*ModelPreference) map[string]*ModelPreference {
	result := make(map[string]*ModelPreference, len(accs))

	for model, acc := range accs {
		// Start from parent prior or uniform prior.
		priorAlpha, priorBeta := 1.0, 1.0
		if parent != nil {
			if p, ok := parent[model]; ok {
				priorAlpha = p.Distribution.Alpha
				priorBeta = p.Distribution.Beta
			}
		}

		pref := &ModelPreference{
			Model: model,
			Distribution: BetaDistribution{
				Alpha: priorAlpha + acc.alpha,
				Beta:  priorBeta + acc.beta,
			},
			TotalInteractions: acc.count,
			LastUpdated:       time.Now().UTC(),
		}

		// If too few observations, keep the parent prior unchanged.
		if acc.count < u.config.MinRecordsPerModel && parent != nil {
			if p, ok := parent[model]; ok {
				pref = p
			}
		}

		result[model] = pref
	}

	// Carry forward parent models that had no new observations.
	for model, p := range parent {
		if _, seen := result[model]; !seen {
			result[model] = p
		}
	}

	return result
}

func (u *OfflineUpdater) finalizeScoped(scoped map[string]map[string]*betaAccumulator, parent map[string]map[string]*ModelPreference) map[string]map[string]*ModelPreference {
	if len(scoped) == 0 && len(parent) == 0 {
		return nil
	}
	result := make(map[string]map[string]*ModelPreference, len(scoped))
	for scope, accs := range scoped {
		var parentScope map[string]*ModelPreference
		if parent != nil {
			parentScope = parent[scope]
		}
		result[scope] = u.finalize(accs, parentScope)
	}
	// Carry forward parent scopes with no new data.
	for scope, prefs := range parent {
		if _, seen := result[scope]; !seen {
			result[scope] = prefs
		}
	}
	return result
}

// CompareVersions produces a summary comparing two policy versions' weight
// distributions for all models that appear in either version.
func CompareVersions(a, b *PolicyVersion) []ModelWeightDiff {
	models := make(map[string]bool)
	for m := range a.Weights.Global {
		models[m] = true
	}
	for m := range b.Weights.Global {
		models[m] = true
	}

	diffs := make([]ModelWeightDiff, 0, len(models))
	for model := range models {
		d := ModelWeightDiff{Model: model}
		if p, ok := a.Weights.Global[model]; ok {
			d.AlphaA = p.Distribution.Alpha
			d.BetaA = p.Distribution.Beta
			d.MeanA = p.Distribution.Mean()
		}
		if p, ok := b.Weights.Global[model]; ok {
			d.AlphaB = p.Distribution.Alpha
			d.BetaB = p.Distribution.Beta
			d.MeanB = p.Distribution.Mean()
		}
		d.MeanDelta = d.MeanB - d.MeanA
		diffs = append(diffs, d)
	}

	sort.Slice(diffs, func(i, j int) bool {
		return diffs[i].Model < diffs[j].Model
	})
	return diffs
}

// ModelWeightDiff summarizes the weight difference for a single model
// between two policy versions.
type ModelWeightDiff struct {
	Model     string  `json:"model"`
	AlphaA    float64 `json:"alpha_a"`
	BetaA     float64 `json:"beta_a"`
	MeanA     float64 `json:"mean_a"`
	AlphaB    float64 `json:"alpha_b"`
	BetaB     float64 `json:"beta_b"`
	MeanB     float64 `json:"mean_b"`
	MeanDelta float64 `json:"mean_delta"`
}

func countFeedback(ds *OfflineDataset) int {
	n := 0
	for _, r := range ds.Records {
		if r.Outcome.Won != nil || r.Outcome.FeedbackType != "" {
			n++
		}
	}
	return n
}

func parentGlobal(pw *PolicyWeights) map[string]*ModelPreference {
	if pw == nil {
		return nil
	}
	return pw.Global
}

func parentCategory(pw *PolicyWeights) map[string]map[string]*ModelPreference {
	if pw == nil {
		return nil
	}
	return pw.Category
}

func parentUser(pw *PolicyWeights) map[string]map[string]*ModelPreference {
	if pw == nil {
		return nil
	}
	return pw.User
}
