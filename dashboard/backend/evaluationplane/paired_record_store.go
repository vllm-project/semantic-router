package evaluationplane

import (
	"fmt"
	"path/filepath"
)

// loadPrivateComparisonRecords reads only server-sealed private evidence. The
// receipt validation binds records.jsonl to the immutable run bundle before
// any statistic is computed.
func (s *Service) loadPrivateComparisonRecords(runID string) ([]executionRecordEvidence, error) {
	checksums, err := s.validatePrivateReceipt(runID)
	if err != nil {
		return nil, err
	}
	if checksums["records.jsonl"] == "" {
		return nil, fmt.Errorf("%w: private records are absent from the sealed receipt", ErrInvalid)
	}
	runDir, err := s.store.checkedRunDir(runID)
	if err != nil {
		return nil, err
	}
	records := make([]executionRecordEvidence, 0)
	seen := make(map[string]bool)
	err = scanEvidenceJSONLines(
		filepath.Join(runDir, "records.jsonl"),
		maxWorkerArtifactBytes,
		maxRecordLineBytes,
		maxRecordsPerRun,
		func(line []byte, lineNumber int) error {
			var record executionRecordEvidence
			if decodeErr := decodeStrictJSONLine(line, &record); decodeErr != nil {
				return fmt.Errorf("%w: private record %d violates the current contract: %w", ErrInvalid, lineNumber, decodeErr)
			}
			if seen[record.ID] {
				return fmt.Errorf("%w: private records contain duplicate id %q", ErrInvalid, record.ID)
			}
			seen[record.ID] = true
			records = append(records, record)
			return nil
		},
	)
	if err != nil {
		return nil, err
	}
	return records, nil
}
