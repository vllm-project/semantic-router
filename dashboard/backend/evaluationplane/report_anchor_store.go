package evaluationplane

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"time"
)

const (
	reportAnchorFileName = "report-anchor.json"
	maxReportAnchorBytes = int64(4 * 1024 * 1024)
)

type reportAnchor struct {
	SchemaVersion              string               `json:"schema_version"`
	AttestationRevision        string               `json:"attestation_revision"`
	RunID                      string               `json:"run_id"`
	ReportDigest               string               `json:"report_digest"`
	ReportSize                 int64                `json:"report_size_bytes"`
	ManifestSemanticDigest     string               `json:"manifest_semantic_digest"`
	ManifestArtifactDigest     string               `json:"manifest_artifact_digest"`
	PrivateReceiptDigest       string               `json:"private_receipt_digest"`
	ExecutionAttestationDigest string               `json:"execution_attestation_digest,omitempty"`
	EvidenceFiles              []sealedEvidenceFile `json:"evidence_files"`
	CreatedAt                  time.Time            `json:"created_at"`
}

func (s *Store) writeReportAnchor(runID string, anchor reportAnchor) error {
	runDir, runDirErr := s.checkedRunDir(runID)
	if runDirErr != nil {
		return runDirErr
	}
	if anchor.SchemaVersion != SchemaVersion || anchor.RunID != runID ||
		anchor.AttestationRevision != ServerAttestationRevision ||
		!digestPattern.MatchString(anchor.ReportDigest) ||
		!digestPattern.MatchString(anchor.ManifestSemanticDigest) ||
		!digestPattern.MatchString(anchor.ManifestArtifactDigest) ||
		!digestPattern.MatchString(anchor.PrivateReceiptDigest) ||
		(anchor.ExecutionAttestationDigest != "" && !digestPattern.MatchString(anchor.ExecutionAttestationDigest)) ||
		anchor.ReportSize < 0 || anchor.CreatedAt.IsZero() {
		return fmt.Errorf("evaluation report anchor is invalid")
	}
	if metadataErr := validateSealedEvidenceMetadata(anchor.EvidenceFiles); metadataErr != nil {
		return metadataErr
	}
	encoded, encodeErr := json.MarshalIndent(anchor, "", "  ")
	if encodeErr != nil {
		return fmt.Errorf("encode evaluation report anchor: %w", encodeErr)
	}
	encoded = append(encoded, '\n')
	path := filepath.Join(runDir, reportAnchorFileName)
	//nolint:gosec // The path is a fixed filename under a validated private run directory.
	file, openErr := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o600)
	if openErr != nil {
		return fmt.Errorf("create evaluation report anchor: %w", openErr)
	}
	remove := true
	defer func() {
		if remove {
			if removeErr := os.Remove(path); removeErr == nil {
				_ = syncEvaluationDirectory(runDir, "evaluation report anchor rollback")
			}
		}
	}()
	_, writeErr := file.Write(encoded)
	if writeErr == nil {
		writeErr = file.Sync()
	}
	closeErr := file.Close()
	if writeErr != nil {
		return fmt.Errorf("write evaluation report anchor: %w", writeErr)
	}
	if closeErr != nil {
		return fmt.Errorf("close evaluation report anchor: %w", closeErr)
	}
	if syncErr := syncEvaluationDirectory(runDir, "evaluation report anchor"); syncErr != nil {
		return syncErr
	}
	remove = false
	return nil
}

func (s *Store) readReportAnchor(runID string) (reportAnchor, error) {
	var anchor reportAnchor
	data, err := s.readReportAnchorBytes(runID)
	if err != nil {
		return reportAnchor{}, fmt.Errorf("read evaluation report anchor: %w", err)
	}
	if err := rejectDuplicateJSONKeys(data); err != nil {
		return reportAnchor{}, fmt.Errorf("decode evaluation report anchor: %w", err)
	}
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&anchor); err != nil {
		return reportAnchor{}, fmt.Errorf("decode evaluation report anchor: %w", err)
	}
	if err := ensureJSONEOF(decoder); err != nil {
		return reportAnchor{}, err
	}
	if anchor.SchemaVersion != SchemaVersion || anchor.RunID != runID ||
		anchor.AttestationRevision != ServerAttestationRevision ||
		!digestPattern.MatchString(anchor.ReportDigest) ||
		!digestPattern.MatchString(anchor.ManifestSemanticDigest) ||
		!digestPattern.MatchString(anchor.ManifestArtifactDigest) ||
		!digestPattern.MatchString(anchor.PrivateReceiptDigest) ||
		(anchor.ExecutionAttestationDigest != "" && !digestPattern.MatchString(anchor.ExecutionAttestationDigest)) ||
		anchor.ReportSize < 0 || anchor.CreatedAt.IsZero() {
		return reportAnchor{}, fmt.Errorf("evaluation report anchor is invalid")
	}
	if err := validateSealedEvidenceMetadata(anchor.EvidenceFiles); err != nil {
		return reportAnchor{}, err
	}
	return anchor, nil
}

func (s *Store) readReportAnchorBytes(runID string) ([]byte, error) {
	runDir, err := s.checkedRunDir(runID)
	if err != nil {
		return nil, err
	}
	return readEvidenceBytes(filepath.Join(runDir, reportAnchorFileName), maxReportAnchorBytes)
}

func (s *Store) reportAnchorDigest(runID string) (string, error) {
	data, err := s.readReportAnchorBytes(runID)
	if err != nil {
		return "", err
	}
	digest, _ := digestAndSize(data)
	return digest, nil
}
