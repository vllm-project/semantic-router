package evaluationplane

import (
	"time"
)

func canonicalizeReportRun(run Run, report *Report, completedAt time.Time) {
	report.Run.ClientRequestID = run.ClientRequestID
	report.Run.Name = run.Name
	report.Run.Description = run.Description
	report.Run.Mixture = copyCatalogMixture(run.Mixture)
	report.Run.TrackEvidenceLevels = copyTrackEvidenceLevels(run.TrackEvidenceLevels)
	report.Run.ControlledPair = copyControlledPairRunMembership(run.ControlledPair)
	report.Run.CreatedAt = run.CreatedAt
	report.Run.StartedAt = copyTime(run.StartedAt)
	report.Run.CompletedAt = copyTime(&completedAt)
	report.Run.Status = StatusCompleted
	report.Run.Error = ""
	report.Run.Progress = RunProgress{
		Percent: 100, Completed: run.Progress.Total, Total: run.Progress.Total, Message: "Evaluation completed",
	}
}

func copyControlledPairRunMembership(source *ControlledPairRunMembership) *ControlledPairRunMembership {
	if source == nil {
		return nil
	}
	copied := *source
	return &copied
}

func copyTrackEvidenceLevels(source map[TrackID]EvidenceLevel) map[TrackID]EvidenceLevel {
	result := make(map[TrackID]EvidenceLevel, len(source))
	for trackID, level := range source {
		result[trackID] = level
	}
	return result
}

func reportRunNameMatches(run Run, reported Run) bool {
	return reported.Name == run.Name
}

func reportRunDescriptionMatches(run Run, reported Run) bool {
	return reported.Description == run.Description
}

func reportRunClientRequestIDMatches(run Run, reported Run) bool {
	return reported.ClientRequestID == run.ClientRequestID
}

func reportRunTimesMatch(run Run, reported Run) bool {
	return sameOptionalTime(run.StartedAt, reported.StartedAt) &&
		sameOptionalTime(run.CompletedAt, reported.CompletedAt)
}

func sameOptionalTime(left, right *time.Time) bool {
	if left == nil || right == nil {
		return left == nil && right == nil
	}
	return left.Equal(*right)
}

func copyTime(value *time.Time) *time.Time {
	if value == nil {
		return nil
	}
	copied := *value
	return &copied
}
