package evaluationplane

import (
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"sort"
	"strings"
	"time"
)

const (
	corruptRunBundleWarningCode = "corrupt_run_bundle"
	maxRunListCursorLength      = 1024
)

type runListWarning struct {
	Code       string
	EvidenceID string
	Message    string
}

type runListCursor struct {
	CreatedAt time.Time `json:"created_at"`
	RunID     string    `json:"run_id"`
}

func (s *Store) ListRuns() ([]Run, error) {
	if err := s.refreshRunIndex(); err != nil {
		return nil, err
	}
	return s.runIndex.allRuns(), nil
}

func (s *Store) listRunsWithinLifecycle() ([]Run, error) {
	if err := s.refreshRunIndexWithinLifecycle(); err != nil {
		return nil, err
	}
	return s.runIndex.allRuns(), nil
}

func (s *Store) refreshRunIndex() error {
	s.lifecycle.mu.Lock()
	defer s.lifecycle.mu.Unlock()
	return s.refreshRunIndexWithinLifecycle()
}

// refreshRunIndexWithinLifecycle follows the root mutation lock order after
// the caller has pinned lifecycle identity against rewrite or deletion.
func (s *Store) refreshRunIndexWithinLifecycle() error {
	s.lifecycle.evidenceMu.Lock()
	defer s.lifecycle.evidenceMu.Unlock()
	s.runIndex.coordinator.Lock()
	defer s.runIndex.coordinator.Unlock()
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.runIndex.hasPendingChanges() {
		return fmt.Errorf("%w: evaluation run ledger publication is incomplete", ErrConflict)
	}
	if err := s.requireNoRunDeletionIntentsUnlocked(); err != nil {
		return err
	}
	return s.refreshRunIndexUnlocked()
}

func (s *Store) refreshRunIndexUnlocked() error {
	runs := make([]Run, 0)
	ownerDigests := make(map[string]string)
	presentRunIDs := make(map[string]bool)
	warnings := make(map[string]runListWarning)
	warningCount := 0
	ownerWarnings := make(map[string]int)
	if err := s.scanRunDirectories(func(evidenceID string, run Run, ownerDigest string, readErr error) {
		if validClientRequestID(evidenceID) {
			presentRunIDs[evidenceID] = true
		}
		if readErr != nil {
			warningCount++
			if ownerDigest != "" {
				ownerWarnings[ownerDigest]++
			}
			if len(warnings) < maxLedgerWarnings {
				warnings[evidenceID] = runListWarning{
					Code: corruptRunBundleWarningCode, EvidenceID: evidenceID, Message: readErr.Error(),
				}
			}
			return
		}
		runs = append(runs, run)
		ownerDigests[run.ID] = ownerDigest
	}); err != nil {
		return err
	}
	s.runIndex.replace(runs, ownerDigests, presentRunIDs, warnings, warningCount, ownerWarnings)
	return nil
}

func (s *Store) listRunLedger(actor Actor, query RunListQuery) (RunLedger, error) {
	cursor, err := decodeRunListCursor(query.Cursor)
	if err != nil {
		return RunLedger{}, err
	}
	runs, totalRuns, indexedWarnings, warningCount, stable := s.runIndex.stablePage(actor, cursor, query.Limit)
	if !stable {
		return RunLedger{}, fmt.Errorf("%w: evaluation run ledger publication is incomplete", ErrConflict)
	}
	publicWarnings := make([]RunLedgerWarning, 0, len(indexedWarnings))
	if actor.administrator {
		for _, warning := range indexedWarnings {
			publicWarnings = append(publicWarnings, publicRunLedgerWarning(warning, true))
		}
	} else if warningCount > 0 {
		publicWarnings = append(publicWarnings, publicRunLedgerWarning(runListWarning{
			Code: corruptRunBundleWarningCode,
		}, false))
	}
	sort.Slice(publicWarnings, func(i, j int) bool {
		return publicWarnings[i].EvidenceID < publicWarnings[j].EvidenceID
	})
	nextCursor := ""
	if len(runs) > query.Limit {
		runs = runs[:query.Limit]
		nextCursor, err = encodeRunListCursor(runs[len(runs)-1])
		if err != nil {
			return RunLedger{}, err
		}
	}
	return RunLedger{
		SchemaVersion: SchemaVersion, Runs: runs, NextCursor: nextCursor, TotalRuns: totalRuns,
		LedgerComplete: warningCount == 0, WarningCount: warningCount, Warnings: publicWarnings,
	}, nil
}

func (s *Store) scanRunDirectories(visit func(string, Run, string, error)) error {
	directory, err := os.Open(s.runsRoot)
	if err != nil {
		return fmt.Errorf("open evaluation runs directory: %w", err)
	}
	defer func() { _ = directory.Close() }()
	for {
		entries, readErr := directory.ReadDir(256)
		for _, entry := range entries {
			if !entry.IsDir() || !validClientRequestID(entry.Name()) {
				visit(quarantinedEvidenceID(entry.Name()), Run{}, "", fmt.Errorf("unexpected entry in evaluation runs directory"))
				continue
			}
			projectedOwner, projected := s.runIndex.ownerDigest(entry.Name())
			run, lifecycle, runErr := s.getRunWithLifecycleUnlocked(entry.Name())
			if errors.Is(runErr, errControlledPairNotCommitted) {
				continue
			}
			ownerDigest := projectedOwner
			if runErr == nil {
				if projected && projectedOwner != lifecycle.OwnerPrincipalDigest {
					runErr = fmt.Errorf("%w: run lifecycle owner identity changed", ErrInvalid)
				} else {
					ownerDigest = lifecycle.OwnerPrincipalDigest
				}
			}
			visit(entry.Name(), run, ownerDigest, runErr)
		}
		if errors.Is(readErr, io.EOF) {
			return nil
		}
		if readErr != nil {
			return fmt.Errorf("list evaluation runs: %w", readErr)
		}
	}
}

func quarantinedEvidenceID(entryName string) string {
	if validClientRequestID(entryName) {
		return entryName
	}
	return digestBytes([]byte(entryName))
}

func runNewer(left, right Run) bool {
	if left.CreatedAt.Equal(right.CreatedAt) {
		return left.ID > right.ID
	}
	return left.CreatedAt.After(right.CreatedAt)
}

func runOlderThanCursor(run Run, cursor runListCursor) bool {
	return run.CreatedAt.Before(cursor.CreatedAt) ||
		(run.CreatedAt.Equal(cursor.CreatedAt) && run.ID < cursor.RunID)
}

func encodeRunListCursor(run Run) (string, error) {
	encoded, err := json.Marshal(runListCursor{CreatedAt: run.CreatedAt, RunID: run.ID})
	if err != nil {
		return "", fmt.Errorf("encode evaluation run cursor: %w", err)
	}
	return base64.RawURLEncoding.EncodeToString(encoded), nil
}

func decodeRunListCursor(raw string) (*runListCursor, error) {
	if raw == "" {
		return nil, nil
	}
	if len(raw) > maxRunListCursorLength {
		return nil, fmt.Errorf("%w: run list cursor is invalid", ErrInvalid)
	}
	data, err := base64.RawURLEncoding.DecodeString(raw)
	if err != nil || len(data) > 512 {
		return nil, fmt.Errorf("%w: run list cursor is invalid", ErrInvalid)
	}
	if err := rejectDuplicateJSONKeys(data); err != nil {
		return nil, fmt.Errorf("%w: run list cursor is invalid", ErrInvalid)
	}
	var cursor runListCursor
	decoder := json.NewDecoder(strings.NewReader(string(data)))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&cursor); err != nil {
		return nil, fmt.Errorf("%w: run list cursor is invalid", ErrInvalid)
	}
	if err := ensureJSONEOF(decoder); err != nil || cursor.CreatedAt.IsZero() || !validClientRequestID(cursor.RunID) {
		return nil, fmt.Errorf("%w: run list cursor is invalid", ErrInvalid)
	}
	return &cursor, nil
}
