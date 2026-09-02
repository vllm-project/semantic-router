package evaluationplane

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"time"
)

const (
	controlledPairManifestFile   = "pair-manifest.json"
	controlledPairMembershipFile = "controlled-pair.json"

	controlledPairStatePublishing = "publishing"
	controlledPairStatePending    = "pending"
	controlledPairStateStarting   = "starting"
	controlledPairStateRunning    = "running"
	controlledPairStateTerminal   = "terminal"
	controlledPairStateCancelling = "cancelling"
	controlledPairStateDeleting   = "deleting"
	controlledPairStateDeleted    = "deleted"
)

var errControlledPairNotCommitted = errors.New("controlled pair publication is not committed")

// controlledPairManifest is the authoritative aggregate for a paired
// execution. Run directories are only projections of this transaction while
// it is publishing or starting; direct and indexed readers consult this fact
// before exposing either member.
type controlledPairManifest struct {
	SchemaVersion                         string     `json:"schema_version"`
	ContractVersion                       string     `json:"contract_version"`
	PairID                                string     `json:"pair_id"`
	ClientRequestID                       string     `json:"client_request_id"`
	Protocol                              string     `json:"protocol"`
	OwnerPrincipalDigest                  string     `json:"owner_principal_digest"`
	BaselineSourceRunID                   string     `json:"baseline_source_run_id"`
	CandidateSourceRunID                  string     `json:"candidate_source_run_id"`
	BaselineRunID                         string     `json:"baseline_run_id"`
	CandidateRunID                        string     `json:"candidate_run_id"`
	BaselineRole                          string     `json:"baseline_role"`
	CandidateRole                         string     `json:"candidate_role"`
	BaselineSourceManifestSemanticDigest  string     `json:"baseline_source_manifest_semantic_digest"`
	CandidateSourceManifestSemanticDigest string     `json:"candidate_source_manifest_semantic_digest"`
	BaselineSourceManifestArtifactDigest  string     `json:"baseline_source_manifest_artifact_digest"`
	CandidateSourceManifestArtifactDigest string     `json:"candidate_source_manifest_artifact_digest"`
	BaselineSourceAnchorDigest            string     `json:"baseline_source_anchor_digest"`
	CandidateSourceAnchorDigest           string     `json:"candidate_source_anchor_digest"`
	BaselineSourceAttestationDigest       string     `json:"baseline_source_attestation_digest"`
	CandidateSourceAttestationDigest      string     `json:"candidate_source_attestation_digest"`
	BaselineMemberManifestDigest          string     `json:"baseline_member_manifest_digest"`
	CandidateMemberManifestDigest         string     `json:"candidate_member_manifest_digest"`
	CohortDigest                          string     `json:"cohort_digest"`
	TreatmentDigest                       string     `json:"treatment_digest"`
	State                                 string     `json:"state"`
	BaselineStageName                     string     `json:"baseline_stage_name,omitempty"`
	CandidateStageName                    string     `json:"candidate_stage_name,omitempty"`
	BaselineRun                           Run        `json:"baseline_run"`
	CandidateRun                          Run        `json:"candidate_run"`
	CreatedAt                             time.Time  `json:"created_at"`
	StartedAt                             *time.Time `json:"started_at,omitempty"`
	StartReceiptDigest                    string     `json:"start_receipt_digest,omitempty"`
	DeletedAt                             *time.Time `json:"deleted_at,omitempty"`
	DeleteReceiptDigest                   string     `json:"delete_receipt_digest,omitempty"`
}

type controlledPairMembership struct {
	SchemaVersion string `json:"schema_version"`
	PairID        string `json:"pair_id"`
	RunID         string `json:"run_id"`
	Role          string `json:"role"`
}

func (s *Store) controlledPairDir(pairID string) (string, error) {
	if !validClientRequestID(pairID) {
		return "", fmt.Errorf("%w: controlled pair identity is invalid", ErrInvalid)
	}
	return filepath.Join(s.controlledPairRoot, pairID), nil
}

func (s *Store) readControlledPair(pairID string) (controlledPairManifest, error) {
	dir, err := s.controlledPairDir(pairID)
	if err != nil {
		return controlledPairManifest{}, err
	}
	if err := requirePrivateDirectory(dir); err != nil {
		if os.IsNotExist(err) {
			return controlledPairManifest{}, fmt.Errorf("%w: controlled pair %s", ErrNotFound, pairID)
		}
		return controlledPairManifest{}, err
	}
	manifestPath := filepath.Join(dir, controlledPairManifestFile)
	var pair controlledPairManifest
	if err := readJSON(manifestPath, &pair); err != nil {
		if !errors.Is(err, ErrNotFound) {
			return controlledPairManifest{}, err
		}
		var tombstone controlledPairTombstone
		if err := readJSON(filepath.Join(dir, controlledPairTombstoneFile), &tombstone); err != nil {
			return controlledPairManifest{}, err
		}
		if err := validateControlledPairTombstone(tombstone); err != nil {
			return controlledPairManifest{}, err
		}
		return controlledPairManifestFromTombstone(tombstone), nil
	}
	if err := validateControlledPairManifest(pair); err != nil {
		return controlledPairManifest{}, err
	}
	return pair, nil
}

func (s *Store) writeControlledPair(pair controlledPairManifest) error {
	if err := validateControlledPairManifest(pair); err != nil {
		return err
	}
	if err := validateControlledPairAggregateReservation(pair); err != nil {
		return err
	}
	dir, err := s.controlledPairDir(pair.PairID)
	if err != nil {
		return err
	}
	created, err := s.pairPersistence.EnsurePrivateDirectory(dir)
	if err != nil {
		return err
	}
	manifestPath := filepath.Join(dir, controlledPairManifestFile)
	needsParentSync := created
	if !created {
		if _, statErr := os.Lstat(manifestPath); os.IsNotExist(statErr) {
			needsParentSync = true
		} else if statErr != nil {
			return statErr
		}
	}
	if needsParentSync {
		if err := s.pairPersistence.SyncDirectory(
			s.controlledPairRoot, "controlled pair directory publication",
		); err != nil {
			return err
		}
	}
	if err := s.pairPersistence.WriteManifest(manifestPath, pair); err != nil {
		return err
	}
	// WriteManifest owns the file rename and pair-directory sync. Explicit
	// SyncDirectory calls are recovery barriers, not a second healthy-path fsync.
	return nil
}

func (s *Store) writeControlledPairDurably(pair controlledPairManifest) error {
	// A readable rename does not prove that the containing directory reached
	// stable storage. In particular, writeJSONAtomic can return after rename
	// when the directory fsync fails. Propagate that uncertainty and let the
	// transaction recovery path decide whether to roll forward or roll back.
	return s.writeControlledPair(pair)
}

func (s *Store) syncControlledPairDirectory(pairID, description string) error {
	dir, err := s.controlledPairDir(pairID)
	if err != nil {
		return err
	}
	if err := s.pairPersistence.SyncDirectory(dir, description); err != nil {
		return fmt.Errorf("controlled pair durability is uncertain: %w", err)
	}
	return nil
}

// syncControlledPairCommitCut closes member-status durability before the
// aggregate manifest. A running or terminal aggregate can therefore never be
// acknowledged while either member transition remains only visibly renamed.
func (s *Store) syncControlledPairCommitCut(pair controlledPairManifest, description string) error {
	for _, runID := range []string{pair.BaselineRunID, pair.CandidateRunID} {
		runDir := filepath.Join(s.runsRoot, runID)
		if err := s.syncRunStatusDirectory(runDir, description+" member status"); err != nil {
			return err
		}
		if err := s.eventPersistence.Sync(
			filepath.Join(runDir, eventsFileName), description+" member event log",
		); err != nil {
			return err
		}
	}
	return s.syncControlledPairDirectory(pair.PairID, description+" aggregate")
}

func validateControlledPairManifest(pair controlledPairManifest) error {
	ids := []string{
		pair.PairID, pair.ClientRequestID, pair.BaselineSourceRunID, pair.CandidateSourceRunID,
		pair.BaselineRunID, pair.CandidateRunID,
	}
	for _, id := range ids {
		if !validClientRequestID(id) {
			return fmt.Errorf("%w: controlled pair manifest identity is invalid", ErrInvalid)
		}
	}
	if pair.SchemaVersion != SchemaVersion || pair.ContractVersion != controlledPairProtocolVersion ||
		pair.PairID != pair.ClientRequestID || pair.Protocol != controlledPairInterleaveABBA ||
		!digestPattern.MatchString(pair.OwnerPrincipalDigest) ||
		pair.BaselineRole != controlledPairRoleBaseline || pair.CandidateRole != controlledPairRoleCandidate ||
		!controlledPairIdentityDigestsValid(pair) ||
		!digestPattern.MatchString(pair.CohortDigest) || !digestPattern.MatchString(pair.TreatmentDigest) ||
		pair.CreatedAt.IsZero() || pair.BaselineRunID != pair.BaselineRun.ID ||
		pair.CandidateRunID != pair.CandidateRun.ID || pair.BaselineRun.BaselineRunID != "" ||
		pair.CandidateRun.BaselineRunID != pair.BaselineRunID ||
		!pair.BaselineRun.CreatedAt.Before(pair.CandidateRun.CreatedAt) ||
		!controlledPairRunMembershipMatches(pair.BaselineRun, pair.PairID, controlledPairRoleBaseline) ||
		!controlledPairRunMembershipMatches(pair.CandidateRun, pair.PairID, controlledPairRoleCandidate) {
		return fmt.Errorf("%w: controlled pair manifest contract is invalid", ErrInvalid)
	}
	seen := make(map[string]bool, len(ids)-1)
	for _, id := range ids[2:] {
		if id == pair.PairID || seen[id] {
			return fmt.Errorf("%w: controlled pair manifest identities must be distinct", ErrInvalid)
		}
		seen[id] = true
	}
	if err := validateStoredRun(pair.BaselineRun.ID, pair.BaselineRun); err != nil {
		return fmt.Errorf("%w: controlled pair baseline snapshot is invalid", ErrInvalid)
	}
	if err := validateStoredRun(pair.CandidateRun.ID, pair.CandidateRun); err != nil {
		return fmt.Errorf("%w: controlled pair candidate snapshot is invalid", ErrInvalid)
	}
	switch pair.State {
	case controlledPairStatePublishing:
		if !stagedRunBundleNamePattern.MatchString(pair.BaselineStageName) ||
			!stagedRunBundleNamePattern.MatchString(pair.CandidateStageName) ||
			pair.BaselineStageName == pair.CandidateStageName || pair.StartedAt != nil ||
			pair.StartReceiptDigest != "" || !pairSnapshotsPending(pair) || pairHasDeleteIntent(pair) {
			return fmt.Errorf("%w: controlled pair publication intent is invalid", ErrInvalid)
		}
	case controlledPairStatePending:
		if pair.BaselineStageName != "" || pair.CandidateStageName != "" || pair.StartedAt != nil ||
			pair.StartReceiptDigest != "" || !pairSnapshotsPending(pair) || pairHasDeleteIntent(pair) {
			return fmt.Errorf("%w: controlled pair pending state is invalid", ErrInvalid)
		}
	case controlledPairStateStarting:
		if pair.BaselineStageName != "" || pair.CandidateStageName != "" || pair.StartedAt == nil ||
			!digestPattern.MatchString(pair.StartReceiptDigest) || !pairSnapshotsPending(pair) ||
			pair.StartReceiptDigest != controlledPairStartReceipt(pair) || pairHasDeleteIntent(pair) {
			return fmt.Errorf("%w: controlled pair start intent is invalid", ErrInvalid)
		}
	case controlledPairStateRunning:
		if pair.BaselineStageName != "" || pair.CandidateStageName != "" || pair.StartedAt == nil ||
			!digestPattern.MatchString(pair.StartReceiptDigest) || !pairSnapshotsRunning(pair) ||
			pair.StartReceiptDigest != controlledPairStartReceipt(pair) || pairHasDeleteIntent(pair) {
			return fmt.Errorf("%w: controlled pair running state is invalid", ErrInvalid)
		}
	case controlledPairStateTerminal:
		if pair.BaselineStageName != "" || pair.CandidateStageName != "" || pair.StartedAt == nil ||
			!digestPattern.MatchString(pair.StartReceiptDigest) || !pairSnapshotsTerminal(pair) ||
			pair.StartReceiptDigest != controlledPairStartReceipt(pair) || pairHasDeleteIntent(pair) {
			return fmt.Errorf("%w: controlled pair terminal state is invalid", ErrInvalid)
		}
	case controlledPairStateCancelling:
		if pair.BaselineStageName != "" || pair.CandidateStageName != "" || pair.StartedAt == nil ||
			!digestPattern.MatchString(pair.StartReceiptDigest) || !pairSnapshotsRunning(pair) ||
			pair.StartReceiptDigest != controlledPairStartReceipt(pair) || pairHasDeleteIntent(pair) {
			return fmt.Errorf("%w: controlled pair cancellation intent is invalid", ErrInvalid)
		}
	case controlledPairStateDeleting, controlledPairStateDeleted:
		if pair.BaselineStageName != "" || pair.CandidateStageName != "" || pair.DeletedAt == nil ||
			!digestPattern.MatchString(pair.DeleteReceiptDigest) ||
			pair.DeleteReceiptDigest != controlledPairDeleteReceipt(pair) ||
			(!pairSnapshotsPending(pair) && !pairSnapshotsTerminal(pair)) {
			return fmt.Errorf("%w: controlled pair deletion state is invalid", ErrInvalid)
		}
	default:
		return fmt.Errorf("%w: controlled pair state is invalid", ErrInvalid)
	}
	return nil
}

func controlledPairRunMembershipMatches(run Run, pairID, role string) bool {
	return run.ControlledPair != nil && run.ControlledPair.PairID == pairID && run.ControlledPair.Role == role
}

func controlledPairIdentityDigestsValid(pair controlledPairManifest) bool {
	for _, digest := range []string{
		pair.BaselineSourceManifestSemanticDigest, pair.CandidateSourceManifestSemanticDigest,
		pair.BaselineSourceManifestArtifactDigest, pair.CandidateSourceManifestArtifactDigest,
		pair.BaselineSourceAnchorDigest, pair.CandidateSourceAnchorDigest,
		pair.BaselineSourceAttestationDigest, pair.CandidateSourceAttestationDigest,
		pair.BaselineMemberManifestDigest, pair.CandidateMemberManifestDigest,
	} {
		if !digestPattern.MatchString(digest) {
			return false
		}
	}
	return true
}

func pairHasDeleteIntent(pair controlledPairManifest) bool {
	return pair.DeletedAt != nil || pair.DeleteReceiptDigest != ""
}

func pairSnapshotsPending(pair controlledPairManifest) bool {
	return pair.BaselineRun.Status == StatusPending && pair.CandidateRun.Status == StatusPending &&
		pair.BaselineRun.StartedAt == nil && pair.CandidateRun.StartedAt == nil
}

func pairSnapshotsRunning(pair controlledPairManifest) bool {
	return pair.BaselineRun.Status == StatusRunning && pair.CandidateRun.Status == StatusRunning &&
		pair.BaselineRun.StartedAt != nil && pair.CandidateRun.StartedAt != nil && pair.StartedAt != nil &&
		pair.BaselineRun.StartedAt.Equal(*pair.StartedAt) && pair.CandidateRun.StartedAt.Equal(*pair.StartedAt)
}

func pairSnapshotsTerminal(pair controlledPairManifest) bool {
	return terminalStatus(pair.BaselineRun.Status) && terminalStatus(pair.CandidateRun.Status) &&
		pair.BaselineRun.StartedAt != nil && pair.CandidateRun.StartedAt != nil && pair.StartedAt != nil &&
		pair.BaselineRun.StartedAt.Equal(*pair.StartedAt) && pair.CandidateRun.StartedAt.Equal(*pair.StartedAt)
}

func controlledPairStartReceipt(pair controlledPairManifest) string {
	if pair.StartedAt == nil {
		return ""
	}
	digest, err := canonicalValueDigest(map[string]any{
		"pair_id": pair.PairID, "protocol": pair.Protocol,
		"baseline_run_id": pair.BaselineRunID, "candidate_run_id": pair.CandidateRunID,
		"started_at": pair.StartedAt.UTC().Format(time.RFC3339Nano),
	})
	if err != nil {
		return ""
	}
	return digest
}

func controlledPairDeleteReceipt(pair controlledPairManifest) string {
	if pair.DeletedAt == nil {
		return ""
	}
	digest, err := canonicalValueDigest(map[string]any{
		"contract_version":        pair.ContractVersion,
		"pair_id":                 pair.PairID,
		"client_request_id":       pair.ClientRequestID,
		"owner_principal_digest":  pair.OwnerPrincipalDigest,
		"baseline_source_run_id":  pair.BaselineSourceRunID,
		"candidate_source_run_id": pair.CandidateSourceRunID,
		"baseline_run_id":         pair.BaselineRunID,
		"candidate_run_id":        pair.CandidateRunID,
		"deleted_at":              pair.DeletedAt.UTC().Format(time.RFC3339Nano),
	})
	if err != nil {
		return ""
	}
	return digest
}

func writeControlledPairMembership(path string, membership controlledPairMembership) error {
	if membership.SchemaVersion != SchemaVersion || !validClientRequestID(membership.PairID) ||
		!validClientRequestID(membership.RunID) ||
		(membership.Role != controlledPairRoleBaseline && membership.Role != controlledPairRoleCandidate) {
		return fmt.Errorf("%w: controlled pair membership is invalid", ErrInvalid)
	}
	return writeJSONAtomic(filepath.Join(path, controlledPairMembershipFile), membership)
}
