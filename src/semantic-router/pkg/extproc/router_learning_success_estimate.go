package extproc

import (
	"strings"
	"time"
)

// Phase 1 (#3480) defines typed success estimates over an immutable in-memory
// snapshot. It does not select a model, introduce apply mode, or treat the
// QualitySeed prior as P(success). Later PRs own those steps:
//   - #3412 observe-only success_constrained selection
//   - #2240 / #2341 / #2346 materialization, calibration, and apply-mode gate

const (
	successEstimateCalibrated   successEstimateStatus = "calibrated"
	successEstimateInsufficient successEstimateStatus = "insufficient_evidence"
	successEstimateUnsupported  successEstimateStatus = "unsupported"
	successEstimateStale        successEstimateStatus = "stale"
	successEstimateConflict     successEstimateStatus = "conflict"

	successEvidenceScopeCohort   = "cohort"
	successEvidenceScopeDecision = "decision"
	successEvidenceScopeTier     = "tier"
	successEvidenceScopeGlobal   = "global"

	successFallbackSeedOnly             = "seed_only"
	successFallbackMissingCalibration   = "missing_calibration_artifact"
	successFallbackStale                = "stale_evidence"
	successFallbackConflictingScopes    = "conflicting_scopes"
	successFallbackUncalibratedSignal   = "signal_not_calibrated_success"
	successFallbackMissingModelRow      = "calibration_row_missing"
	successFallbackSparseNoBroaderScope = "sparse_scope"
)

type successEstimateStatus string

type successEstimate struct {
	CandidateModel     string
	Status             successEstimateStatus
	Probability        float64
	Uncertainty        float64
	Coverage           float64
	SampleCount        int
	EvidenceScope      string
	FreshnessSeconds   int64
	CalibrationVersion string
	FallbackReason     string
}

type successEstimateConfig struct {
	Now        time.Time
	StaleAfter time.Duration
	Artifact   *successCalibrationArtifact
}

// successCalibrationArtifact is the request-path view of an accepted
// calibration artifact. Phase 1 production wiring passes nil; tests inject
// rows so conservative rules can be exercised without #2341.
type successCalibrationArtifact struct {
	Version string
	Models  map[string]successCalibrationRow
}

type successCalibrationRow struct {
	Probability float64
	Uncertainty float64
	Coverage    float64
}

func estimateCandidateSuccess(
	snap routerLearningEvidenceSnapshot,
	models []string,
	cfg successEstimateConfig,
) []successEstimate {
	if len(models) == 0 {
		return nil
	}
	now := cfg.Now
	if now.IsZero() {
		if !snap.takenAt.IsZero() {
			now = snap.takenAt
		} else {
			now = time.Now().UTC()
		}
	}
	out := make([]successEstimate, 0, len(models))
	for _, model := range models {
		model = strings.TrimSpace(model)
		if model == "" {
			continue
		}
		out = append(out, estimateOneCandidateSuccess(snap, model, successEstimateConfig{
			Now:        now,
			StaleAfter: cfg.StaleAfter,
			Artifact:   cfg.Artifact,
		}))
	}
	return out
}

func estimateOneCandidateSuccess(
	snap routerLearningEvidenceSnapshot,
	model string,
	cfg successEstimateConfig,
) successEstimate {
	exp, scope, found := snap.resolveExperience(model)
	est := successEstimate{
		CandidateModel:   model,
		EvidenceScope:    scope,
		SampleCount:      exp.outcomeSamples(),
		FreshnessSeconds: experienceFreshnessSeconds(exp, cfg.Now),
	}

	if isExperienceStale(exp, cfg.Now, cfg.StaleAfter) {
		return unsupportedOrConflictEstimate(est, successEstimateStale, successFallbackStale)
	}
	if !found || exp.seedOnly() {
		reason := successFallbackSeedOnly
		if found && exp.seedOnly() {
			reason = successFallbackSparseNoBroaderScope
		}
		return unsupportedOrConflictEstimate(est, successEstimateInsufficient, reason)
	}
	if cfg.Artifact == nil || strings.TrimSpace(cfg.Artifact.Version) == "" {
		return unsupportedOrConflictEstimate(est, successEstimateUnsupported, successFallbackMissingCalibration)
	}
	row, ok := cfg.Artifact.Models[model]
	if !ok {
		est.CalibrationVersion = strings.TrimSpace(cfg.Artifact.Version)
		return unsupportedOrConflictEstimate(est, successEstimateUnsupported, successFallbackMissingModelRow)
	}

	est.Status = successEstimateCalibrated
	est.Probability = clamp01(row.Probability)
	est.Uncertainty = clamp01(row.Uncertainty)
	est.Coverage = clamp01(row.Coverage)
	est.CalibrationVersion = strings.TrimSpace(cfg.Artifact.Version)
	return est
}

func unsupportedOrConflictEstimate(
	est successEstimate,
	status successEstimateStatus,
	reason string,
) successEstimate {
	est.Status = status
	est.Probability = 0
	est.Uncertainty = 0
	est.Coverage = 0
	est.FallbackReason = reason
	if status != successEstimateCalibrated {
		// Keep calibration identity only as audit context, never as a gate.
		if status != successEstimateUnsupported {
			est.CalibrationVersion = ""
		}
	}
	return est
}

func successEstimateFromUncalibratedSignal(model string, signalName string, _ float64) successEstimate {
	return successEstimate{
		CandidateModel: strings.TrimSpace(model),
		Status:         successEstimateUnsupported,
		FallbackReason: successFallbackUncalibratedSignal + ":" + strings.TrimSpace(signalName),
	}
}

func (e routerLearningModelExperience) outcomeSamples() int {
	return e.GoodFitCount + e.UnderpoweredCount + e.OverprovisionedCount + e.FailedCount
}

func (e routerLearningModelExperience) seedOnly() bool {
	return e.outcomeSamples() == 0
}

func experienceFreshnessSeconds(exp routerLearningModelExperience, now time.Time) int64 {
	if exp.LastUpdated.IsZero() || now.IsZero() {
		return 0
	}
	age := now.Sub(exp.LastUpdated)
	if age < 0 {
		return 0
	}
	return int64(age.Seconds())
}

func isExperienceStale(exp routerLearningModelExperience, now time.Time, staleAfter time.Duration) bool {
	if staleAfter <= 0 || exp.LastUpdated.IsZero() || now.IsZero() {
		return false
	}
	return now.Sub(exp.LastUpdated) > staleAfter
}

func scopedExperienceRatesConflict(a routerLearningModelExperience, b routerLearningModelExperience) bool {
	aRate := goodFitRate(a)
	bRate := goodFitRate(b)
	delta := aRate - bRate
	if delta < 0 {
		delta = -delta
	}
	// Only an explicit, large disagreement is a conflict. Ordinary hierarchical
	// backoff (sparse specific scope, populated broader scope) is not a merge.
	return delta > 0.5
}

func goodFitRate(exp routerLearningModelExperience) float64 {
	total := float64(exp.outcomeSamples())
	if total <= 0 {
		return 0
	}
	return float64(exp.GoodFitCount) / total
}
