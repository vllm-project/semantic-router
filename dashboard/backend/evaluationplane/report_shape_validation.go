package evaluationplane

import "fmt"

// validateReportShape covers fields shared by an untrusted worker draft and a
// sealed public report. Publication-only fields have dedicated validators.
func validateReportShape(runID string, report Report) error {
	if report.SchemaVersion != SchemaVersion {
		return fmt.Errorf("evaluation report schema_version must be %q", SchemaVersion)
	}
	if report.Run.SchemaVersion != SchemaVersion || report.Provenance.SchemaVersion != SchemaVersion {
		return fmt.Errorf("evaluation report nested schema_version must be %q", SchemaVersion)
	}
	if report.Run.ID != runID || report.Run.ClientRequestID != runID {
		return fmt.Errorf("evaluation report run identity mismatch")
	}
	if !validChangeProfile(report.Run.ChangeProfile) {
		return fmt.Errorf("evaluation report change_profile is invalid")
	}
	if report.Provenance.TargetID != report.Run.TargetID || report.Provenance.Seed != report.Run.Seed {
		return fmt.Errorf("evaluation report provenance identity mismatch")
	}
	if !validDecisionVerdict(report.Summary.Verdict) {
		return fmt.Errorf("evaluation report summary verdict is invalid")
	}
	if report.Run.SuiteIDs == nil || report.Run.TrackIDs == nil || report.Tracks == nil ||
		report.Metrics == nil || report.Gates == nil || report.Recommendations == nil ||
		report.Artifacts == nil || report.MethodReports == nil {
		return fmt.Errorf("evaluation report required collections cannot be null")
	}
	for _, track := range report.Tracks {
		if track.Metrics == nil || track.Gates == nil {
			return fmt.Errorf("evaluation track report required collections cannot be null")
		}
		for _, gate := range track.Gates {
			if err := validateReportGate(gate, report.Run.ChangeProfile); err != nil {
				return err
			}
		}
	}
	if err := validateReportMetrics(report.Metrics, report.Run.TrackIDs); err != nil {
		return err
	}
	for _, gate := range report.Gates {
		if err := validateReportGate(gate, report.Run.ChangeProfile); err != nil {
			return err
		}
	}
	return nil
}
