package evaluationplane

import (
	"fmt"
	"math"
	"strings"
)

var safeIDPattern = portableIDPattern

func validateStoredRun(bundleID string, run Run) error {
	if run.SchemaVersion != SchemaVersion {
		return fmt.Errorf("status schema_version must be %q", SchemaVersion)
	}
	if run.ID != bundleID || run.ClientRequestID != run.ID || !validClientRequestID(run.ID) {
		return fmt.Errorf("status run identity does not match its bundle")
	}
	if run.Name != strings.TrimSpace(run.Name) || run.Name == "" || len(run.Name) > maxRunNameLength ||
		run.Description != strings.TrimSpace(run.Description) || len(run.Description) > maxRunDescriptionLength {
		return fmt.Errorf("status run metadata is invalid")
	}
	if (run.Mode != ModeReplay && run.Mode != ModeLive) || evidenceLevelRank(run.EvidenceLevel) < 0 ||
		!validChangeProfile(run.ChangeProfile) || !safeIDPattern.MatchString(run.TargetID) {
		return fmt.Errorf("status run evaluation identity is invalid")
	}
	if run.Mixture != nil {
		if !targetIDMatchesMixture(run.TargetID, run.Mixture.ID) {
			return fmt.Errorf("mixture status must freeze its selected subject")
		}
		if err := validateMixtureContract(manifestMixtureFromCatalog(run.Mixture)); err != nil {
			return fmt.Errorf("status mixture is invalid: %w", err)
		}
	} else if run.Mode == ModeLive {
		return fmt.Errorf("live mixture status must freeze its selected subject")
	}
	if !validStoredSuiteIDs(run.SuiteIDs) || !validStoredTrackIDs(run.TrackIDs) ||
		run.SampleLimit < 1 || run.SampleLimit > maxSampleLimit ||
		run.Concurrency < 1 || run.Concurrency > maxRunConcurrency ||
		run.Seed < 0 || run.Seed > 1<<32-1 || run.CreatedAt.IsZero() {
		return fmt.Errorf("status run execution contract is invalid")
	}
	if !validStoredTrackEvidenceLevels(run.TrackIDs, run.TrackEvidenceLevels, run.EvidenceLevel) {
		return fmt.Errorf("status run track evidence levels are invalid")
	}
	if run.BaselineRunID != "" && !validClientRequestID(run.BaselineRunID) {
		return fmt.Errorf("status baseline identity is invalid")
	}
	if run.ControlledPair != nil && (!validClientRequestID(run.ControlledPair.PairID) ||
		(run.ControlledPair.Role != controlledPairRoleBaseline && run.ControlledPair.Role != controlledPairRoleCandidate)) {
		return fmt.Errorf("status controlled pair membership is invalid")
	}
	if err := validateCapacityRunContract(
		run.Mode,
		run.TrackIDs,
		run.Concurrency,
		run.CapacitySLO,
		run.CapacityLoadProtocol,
	); err != nil {
		return fmt.Errorf("status capacity SLO is invalid: %w", err)
	}
	if math.IsNaN(run.Progress.Percent) || math.IsInf(run.Progress.Percent, 0) ||
		run.Progress.Percent < 0 || run.Progress.Percent > 100 ||
		run.Progress.Total != len(run.TrackIDs) || run.Progress.Completed < 0 ||
		run.Progress.Completed > run.Progress.Total ||
		(run.Progress.CurrentTrackID != "" && !containsTrack(run.TrackIDs, run.Progress.CurrentTrackID)) {
		return fmt.Errorf("status run progress is invalid")
	}
	if run.Progress.Message == "" || run.Progress.Message != strings.TrimSpace(run.Progress.Message) ||
		len(run.Progress.Message) > maxWorkerMessageBytes {
		return fmt.Errorf("status run progress message is invalid")
	}
	return validateStoredRunState(run)
}

func targetIDMatchesMixture(targetID, mixtureID string) bool {
	if targetID == mixtureID {
		return true
	}
	suffix := "--" + mixtureID
	if !strings.HasSuffix(targetID, suffix) {
		return false
	}
	return deploymentIDPattern.MatchString(strings.TrimSuffix(targetID, suffix))
}

func validStoredTrackEvidenceLevels(
	trackIDs []TrackID,
	levels map[TrackID]EvidenceLevel,
	headline EvidenceLevel,
) bool {
	if len(levels) != len(trackIDs) {
		return false
	}
	for _, trackID := range trackIDs {
		level, present := levels[trackID]
		if !present || evidenceLevelRank(level) < 0 {
			return false
		}
	}
	return headline == runEvidenceHeadline(levels, trackIDs)
}

func runEvidenceHeadline(levels map[TrackID]EvidenceLevel, trackIDs []TrackID) EvidenceLevel {
	if len(trackIDs) == 0 {
		return "E0"
	}
	weakest := levels[trackIDs[0]]
	for _, trackID := range trackIDs[1:] {
		if evidenceLevelRank(levels[trackID]) < evidenceLevelRank(weakest) {
			weakest = levels[trackID]
		}
	}
	return weakest
}

func validateStoredRunState(run Run) error {
	if run.StartedAt != nil && run.StartedAt.Before(run.CreatedAt) {
		return fmt.Errorf("status started_at predates created_at")
	}
	if run.CompletedAt != nil && (run.CompletedAt.Before(run.CreatedAt) ||
		(run.StartedAt != nil && run.CompletedAt.Before(*run.StartedAt))) {
		return fmt.Errorf("status completed_at is outside the run lifetime")
	}
	if run.Error != strings.TrimSpace(run.Error) || len(run.Error) > maxWorkerMessageBytes {
		return fmt.Errorf("status error is invalid")
	}
	switch run.Status {
	case StatusPending:
		if run.StartedAt != nil || run.CompletedAt != nil || run.Error != "" ||
			run.Progress.Percent != 0 || run.Progress.Completed != 0 || run.Progress.CurrentTrackID != "" {
			return fmt.Errorf("pending status contains execution state")
		}
	case StatusRunning:
		if run.StartedAt == nil || run.CompletedAt != nil || run.Error != "" {
			return fmt.Errorf("running status timestamps or error are inconsistent")
		}
	case StatusSealing:
		if run.StartedAt == nil || run.CompletedAt != nil || run.Error != "" {
			return fmt.Errorf("sealing status timestamps or error are inconsistent")
		}
	case StatusCompleted:
		if run.StartedAt == nil || run.CompletedAt == nil || run.Error != "" ||
			run.Progress.Percent != 100 || run.Progress.Completed != run.Progress.Total || run.Progress.CurrentTrackID != "" {
			return fmt.Errorf("completed status is not terminal and successful")
		}
	case StatusFailed:
		if run.StartedAt == nil || run.CompletedAt == nil || run.Error == "" {
			return fmt.Errorf("failed status lacks execution timestamps or error")
		}
	case StatusCancelled:
		if run.CompletedAt == nil || run.Error != "" {
			return fmt.Errorf("cancelled status timestamps or error are inconsistent")
		}
	default:
		return fmt.Errorf("status contains an invalid run state")
	}
	return nil
}

func validStoredSuiteIDs(values []string) bool {
	if len(values) == 0 {
		return false
	}
	seen := make(map[string]bool, len(values))
	for _, value := range values {
		if !safeIDPattern.MatchString(value) || seen[value] {
			return false
		}
		seen[value] = true
	}
	canonical := canonicalSuiteIDs(values)
	if len(canonical) != len(values) {
		return false
	}
	for index := range values {
		if values[index] != canonical[index] {
			return false
		}
	}
	return true
}

func validStoredTrackIDs(values []TrackID) bool {
	if len(values) == 0 {
		return false
	}
	seen := make(map[TrackID]bool, len(values))
	for _, value := range values {
		if !containsTrack(allTrackIDs, value) || seen[value] {
			return false
		}
		seen[value] = true
	}
	canonical := canonicalTrackIDs(values)
	for index := range values {
		if values[index] != canonical[index] {
			return false
		}
	}
	return true
}

func validateResourceID(id string) error {
	if !validClientRequestID(id) {
		return fmt.Errorf("%w: evaluation run id must be a canonical UUID", ErrInvalid)
	}
	return nil
}
