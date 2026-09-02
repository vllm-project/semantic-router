package evaluationplane

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strings"
)

func (s *Service) ReportJSONAs(actor Actor, runID string) ([]byte, error) {
	releaseOperation, operationErr := s.beginOperation()
	if operationErr != nil {
		return nil, operationErr
	}
	defer releaseOperation()
	release, err := s.acquireAuthorizedEvidenceRead(actor, runID)
	if err != nil {
		return nil, err
	}
	defer release()
	return s.reportJSONVerified(runID)
}

func (s *Service) reportJSONVerified(runID string) ([]byte, error) {
	run, err := s.store.GetRun(runID)
	if err != nil {
		return nil, err
	}
	if run.Status != StatusCompleted {
		return nil, fmt.Errorf("%w: evaluation report is available only for completed runs", ErrConflict)
	}
	data, err := s.store.ReadReport(runID)
	if err != nil {
		return nil, err
	}
	report, err := decodeReportStrict(runID, data)
	if err != nil {
		return nil, err
	}
	manifest, _, err := s.readDurableManifest(runID)
	if err != nil {
		return nil, err
	}
	if err := validateReportFrozenFields(run, manifest, report); err != nil {
		return nil, err
	}
	if err := s.verifyReportAnchor(runID, data, report.AttestationRevision); err != nil {
		return nil, err
	}
	if err := s.rejectConfiguredSecretBytes(data); err != nil {
		return nil, err
	}
	receipt, ok := findArtifactByName(report, publicChecksumArtifactName)
	if !ok {
		return nil, fmt.Errorf("%w: public artifact checksum receipt is unavailable", ErrInvalid)
	}
	if err := s.verifyPublicChecksum(runID, report, receipt); err != nil {
		return nil, err
	}
	return data, nil
}

func decodeReportStrict(runID string, data []byte) (Report, error) {
	if err := rejectDuplicateJSONKeys(data); err != nil {
		return Report{}, fmt.Errorf("%w: decode evaluation report: %w", ErrInvalid, err)
	}
	var report Report
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&report); err != nil {
		return Report{}, fmt.Errorf("%w: decode evaluation report: %w", ErrInvalid, err)
	}
	if err := ensureJSONEOF(decoder); err != nil {
		return Report{}, fmt.Errorf("%w: %w", ErrInvalid, err)
	}
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(data, &fields); err != nil {
		return Report{}, fmt.Errorf("%w: decode evaluation report fields: %w", ErrInvalid, err)
	}
	if _, present := fields["routing_recipe_report"]; !present {
		return Report{}, fmt.Errorf("%w: evaluation report omits its server-owned routing_recipe_report field", ErrInvalid)
	}
	if report.AttestationRevision != ServerAttestationRevision {
		return Report{}, fmt.Errorf("%w: evaluation report attestation_revision must be %q", ErrInvalid, ServerAttestationRevision)
	}
	if err := validateReportShape(runID, report); err != nil {
		return Report{}, fmt.Errorf("%w: %w", ErrInvalid, err)
	}
	if err := validatePublishedRoutingRecipeReportShape(report); err != nil {
		return Report{}, fmt.Errorf("%w: %w", ErrInvalid, err)
	}
	return report, nil
}

func (s *Service) decodedReport(runID string) (Report, error) {
	data, err := s.reportJSONVerified(runID)
	if err != nil {
		return Report{}, err
	}
	var report Report
	if err := json.Unmarshal(data, &report); err != nil {
		return Report{}, fmt.Errorf("decode evaluation report: %w", err)
	}
	return report, nil
}

func validMetricDirection(direction string) bool {
	switch direction {
	case "", "higher_is_better", "lower_is_better", "target":
		return true
	default:
		return false
	}
}

func validateReportGate(gate Gate, profile ChangeProfile) error {
	if gate.ChangeProfile != profile {
		return fmt.Errorf("evaluation gate %q change_profile does not match its run", gate.ID)
	}
	if gate.ContractVersion != GateContractVersion {
		return fmt.Errorf("evaluation gate %q contract_version must be %q", gate.ID, GateContractVersion)
	}
	if len(gate.EvidenceRefs) == 0 {
		return fmt.Errorf("evaluation gate %q evidence_refs must contain at least one reference", gate.ID)
	}
	for _, ref := range gate.EvidenceRefs {
		if strings.TrimSpace(ref) == "" {
			return fmt.Errorf("evaluation gate %q evidence_refs must be non-empty", gate.ID)
		}
	}
	if gate.SampleCount != nil && *gate.SampleCount < 0 {
		return fmt.Errorf("evaluation gate %q sample_count cannot be negative", gate.ID)
	}
	if gate.Coverage != nil && (gate.Coverage.Evaluated < 0 || gate.Coverage.Total < 0 ||
		gate.Coverage.Unavailable < 0 || gate.Coverage.Fraction < 0 || gate.Coverage.Fraction > 1) {
		return fmt.Errorf("evaluation gate %q coverage is invalid", gate.ID)
	}
	if !validGateDisposition(gate.Disposition) {
		return fmt.Errorf("evaluation gate %q disposition is invalid", gate.ID)
	}
	if !validGateVerdict(gate.Verdict) {
		return fmt.Errorf("evaluation gate %q verdict is invalid", gate.ID)
	}
	return nil
}
