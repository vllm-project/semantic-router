package evaluationplane

import "fmt"

const (
	defaultRunPageLimit = 50
	maxRunPageLimit     = 200
	maxLedgerWarnings   = 100
)

const quarantinedRunMessage = "Durable run status evidence is unreadable or invalid and has been quarantined."

var redactedRunLedgerEvidenceID = digestString("evaluation-run-ledger:redacted-evidence")

// RunLedgerWarning is a public, path-safe description of durable run evidence
// that could not be projected into the readable run list. Detailed parse and
// filesystem diagnostics remain in the server log.
type RunLedgerWarning struct {
	Code         string `json:"code"`
	EvidenceID   string `json:"evidence_id"`
	EvidenceFile string `json:"evidence_file"`
	Message      string `json:"message"`
}

// RunLedger is an atomic actor-scoped view of readable runs and attributable
// quarantined durable entries. Administrators receive the global projection;
// ordinary principals never observe another owner's or an unattributable
// quarantine count. Scientific decisions separately require the complete
// administrator projection.
type RunLedger struct {
	SchemaVersion  string             `json:"schema_version"`
	Runs           []Run              `json:"runs"`
	NextCursor     string             `json:"next_cursor,omitempty"`
	TotalRuns      int                `json:"total_runs"`
	LedgerComplete bool               `json:"ledger_complete"`
	WarningCount   int                `json:"warning_count"`
	Warnings       []RunLedgerWarning `json:"warnings"`
}

type RunListQuery struct {
	Limit  int
	Cursor string
}

func publicRunLedgerWarning(warning runListWarning, revealEvidenceID bool) RunLedgerWarning {
	evidenceID := redactedRunLedgerEvidenceID
	if revealEvidenceID {
		evidenceID = warning.EvidenceID
	}
	return RunLedgerWarning{
		Code:         warning.Code,
		EvidenceID:   evidenceID,
		EvidenceFile: runFileName,
		Message:      quarantinedRunMessage,
	}
}

func (s *Service) ListRunLedgerPageAs(actor Actor, query RunListQuery) (RunLedger, error) {
	release, err := s.beginOperation()
	if err != nil {
		return RunLedger{}, err
	}
	defer release()
	if err := validateActor(actor); err != nil {
		return RunLedger{}, err
	}
	if query.Limit == 0 {
		query.Limit = defaultRunPageLimit
	}
	if query.Limit < 1 || query.Limit > maxRunPageLimit {
		return RunLedger{}, fmt.Errorf("%w: run list limit must be between 1 and %d", ErrInvalid, maxRunPageLimit)
	}
	return s.store.listRunLedger(actor, query)
}

func (s *Service) requireCompleteRunLedger() error {
	// Scientific decisions re-derive the projection from canonical evidence so
	// out-of-band corruption cannot be hidden by a previously healthy snapshot.
	// Polling list requests use the maintained index and remain O(page).
	if err := s.store.refreshRunIndex(); err != nil {
		return err
	}
	return s.requireCompleteRunLedgerProjection()
}

// requireCompleteRunLedgerWithinLifecycle is for operations that already hold
// lifecycle.mu and therefore must enter the remaining refresh locks directly.
func (s *Service) requireCompleteRunLedgerWithinLifecycle() error {
	if err := s.store.refreshRunIndexWithinLifecycle(); err != nil {
		return err
	}
	return s.requireCompleteRunLedgerProjection()
}

func (s *Service) requireCompleteRunLedgerProjection() error {
	ledger, err := s.store.listRunLedger(SystemActor(), RunListQuery{Limit: 1})
	if err != nil {
		return err
	}
	if !ledger.LedgerComplete {
		return fmt.Errorf(
			"%w: evaluation run ledger is incomplete (%d quarantined run bundle(s)); repair the durable evidence before selecting a baseline or comparing runs",
			ErrConflict,
			ledger.WarningCount,
		)
	}
	return nil
}
