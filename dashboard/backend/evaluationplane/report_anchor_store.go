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
	SchemaVersion        string               `json:"schema_version"`
	RunID                string               `json:"run_id"`
	ReportDigest         string               `json:"report_digest"`
	ReportSize           int64                `json:"report_size_bytes"`
	ManifestDigest       string               `json:"manifest_digest"`
	PrivateReceiptDigest string               `json:"private_receipt_digest"`
	EvidenceFiles        []sealedEvidenceFile `json:"evidence_files"`
	CreatedAt            time.Time            `json:"created_at"`
}

func (s *Store) writeReportAnchor(runID string, anchor reportAnchor) error {
	runDir, err := s.checkedRunDir(runID)
	if err != nil {
		return err
	}
	if anchor.SchemaVersion != SchemaVersion || anchor.RunID != runID ||
		!digestPattern.MatchString(anchor.ReportDigest) || !digestPattern.MatchString(anchor.ManifestDigest) ||
		!digestPattern.MatchString(anchor.PrivateReceiptDigest) || anchor.ReportSize < 0 || anchor.CreatedAt.IsZero() {
		return fmt.Errorf("evaluation report anchor is invalid")
	}
	if metadataErr := validateSealedEvidenceMetadata(anchor.EvidenceFiles); metadataErr != nil {
		return metadataErr
	}
	encoded, err := json.MarshalIndent(anchor, "", "  ")
	if err != nil {
		return fmt.Errorf("encode evaluation report anchor: %w", err)
	}
	encoded = append(encoded, '\n')
	path := filepath.Join(runDir, reportAnchorFileName)
	//nolint:gosec // The path is a fixed filename under a validated private run directory.
	file, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o600)
	if err != nil {
		return fmt.Errorf("create evaluation report anchor: %w", err)
	}
	remove := true
	defer func() {
		if remove {
			_ = os.Remove(path)
		}
	}()
	if _, err = file.Write(encoded); err == nil {
		err = file.Sync()
	}
	closeErr := file.Close()
	if err != nil {
		return fmt.Errorf("write evaluation report anchor: %w", err)
	}
	if closeErr != nil {
		return fmt.Errorf("close evaluation report anchor: %w", closeErr)
	}
	remove = false
	return nil
}

func (s *Store) readReportAnchor(runID string) (reportAnchor, error) {
	runDir, err := s.checkedRunDir(runID)
	if err != nil {
		return reportAnchor{}, err
	}
	var anchor reportAnchor
	data, err := readEvidenceBytes(filepath.Join(runDir, reportAnchorFileName), maxReportAnchorBytes)
	if err != nil {
		return reportAnchor{}, fmt.Errorf("read evaluation report anchor: %w", err)
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
		!digestPattern.MatchString(anchor.ReportDigest) || !digestPattern.MatchString(anchor.ManifestDigest) ||
		!digestPattern.MatchString(anchor.PrivateReceiptDigest) ||
		anchor.ReportSize < 0 || anchor.CreatedAt.IsZero() {
		return reportAnchor{}, fmt.Errorf("evaluation report anchor is invalid")
	}
	if err := validateSealedEvidenceMetadata(anchor.EvidenceFiles); err != nil {
		return reportAnchor{}, err
	}
	return anchor, nil
}
