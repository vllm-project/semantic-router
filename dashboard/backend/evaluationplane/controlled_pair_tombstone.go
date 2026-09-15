package evaluationplane

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"time"
)

const controlledPairTombstoneFile = "pair-tombstone.json"

type controlledPairTombstone struct {
	SchemaVersion        string     `json:"schema_version"`
	ContractVersion      string     `json:"contract_version"`
	PairID               string     `json:"pair_id"`
	ClientRequestID      string     `json:"client_request_id"`
	OwnerPrincipalDigest string     `json:"owner_principal_digest"`
	BaselineSourceRunID  string     `json:"baseline_source_run_id"`
	CandidateSourceRunID string     `json:"candidate_source_run_id"`
	BaselineRunID        string     `json:"baseline_run_id"`
	CandidateRunID       string     `json:"candidate_run_id"`
	DeletedAt            *time.Time `json:"deleted_at"`
	DeleteReceiptDigest  string     `json:"delete_receipt_digest"`
}

func controlledPairTombstoneFromManifest(pair controlledPairManifest) controlledPairTombstone {
	return controlledPairTombstone{
		SchemaVersion: SchemaVersion, ContractVersion: controlledPairProtocolVersion,
		PairID: pair.PairID, ClientRequestID: pair.ClientRequestID,
		OwnerPrincipalDigest: pair.OwnerPrincipalDigest,
		BaselineSourceRunID:  pair.BaselineSourceRunID, CandidateSourceRunID: pair.CandidateSourceRunID,
		BaselineRunID: pair.BaselineRunID, CandidateRunID: pair.CandidateRunID,
		DeletedAt: pair.DeletedAt, DeleteReceiptDigest: pair.DeleteReceiptDigest,
	}
}

// controlledPairDeletionTime keeps the RFC3339Nano representation at the
// reserved six fractional digits. time.Time JSON trims trailing zeroes; a
// non-zero final microsecond makes collection's pre-commit byte estimate equal
// the durable identity tombstone instead of varying by clock value.
func controlledPairDeletionTime(now time.Time) time.Time {
	deletedAt := now.UTC().Truncate(time.Microsecond)
	if deletedAt.Nanosecond()/int(time.Microsecond)%10 == 0 {
		deletedAt = deletedAt.Add(time.Microsecond)
	}
	return deletedAt
}

func controlledPairTombstoneBytes(pair controlledPairManifest) (int64, error) {
	projection := pair
	if projection.DeletedAt == nil {
		deletedAt := time.Unix(1_999_999_999, 999_999_000).UTC()
		projection.DeletedAt = &deletedAt
	}
	if projection.DeleteReceiptDigest == "" {
		projection.DeleteReceiptDigest = digestString("controlled-pair-tombstone-reservation")
	}
	encoded, err := json.MarshalIndent(controlledPairTombstoneFromManifest(projection), "", "  ")
	if err != nil {
		return 0, err
	}
	return int64(len(encoded) + 1), nil
}

func validateControlledPairTombstone(tombstone controlledPairTombstone) error {
	ids := []string{
		tombstone.PairID, tombstone.ClientRequestID, tombstone.BaselineSourceRunID,
		tombstone.CandidateSourceRunID, tombstone.BaselineRunID, tombstone.CandidateRunID,
	}
	for _, id := range ids {
		if !validClientRequestID(id) {
			return fmt.Errorf("%w: controlled pair tombstone identity is invalid", ErrInvalid)
		}
	}
	seen := map[string]bool{tombstone.PairID: true}
	for _, id := range ids[2:] {
		if seen[id] {
			return fmt.Errorf("%w: controlled pair tombstone identities must be distinct", ErrInvalid)
		}
		seen[id] = true
	}
	if tombstone.SchemaVersion != SchemaVersion || tombstone.ContractVersion != controlledPairProtocolVersion ||
		tombstone.PairID != tombstone.ClientRequestID ||
		!digestPattern.MatchString(tombstone.OwnerPrincipalDigest) || tombstone.DeletedAt == nil ||
		!digestPattern.MatchString(tombstone.DeleteReceiptDigest) ||
		tombstone.DeleteReceiptDigest != controlledPairDeleteReceipt(controlledPairManifestFromTombstone(tombstone)) {
		return fmt.Errorf("%w: controlled pair tombstone contract is invalid", ErrInvalid)
	}
	return nil
}

func controlledPairManifestFromTombstone(tombstone controlledPairTombstone) controlledPairManifest {
	return controlledPairManifest{
		SchemaVersion: tombstone.SchemaVersion, ContractVersion: tombstone.ContractVersion,
		PairID: tombstone.PairID, ClientRequestID: tombstone.ClientRequestID,
		OwnerPrincipalDigest: tombstone.OwnerPrincipalDigest,
		BaselineSourceRunID:  tombstone.BaselineSourceRunID, CandidateSourceRunID: tombstone.CandidateSourceRunID,
		BaselineRunID: tombstone.BaselineRunID, CandidateRunID: tombstone.CandidateRunID,
		State: controlledPairStateDeleted, DeletedAt: tombstone.DeletedAt,
		DeleteReceiptDigest: tombstone.DeleteReceiptDigest,
	}
}

func (s *Store) writeControlledPairTombstoneDurably(pair controlledPairManifest) error {
	tombstone := controlledPairTombstoneFromManifest(pair)
	if err := validateControlledPairTombstone(tombstone); err != nil {
		return err
	}
	reservation, err := controlledPairIntentReservationBytes(pair)
	if err != nil {
		return err
	}
	encoded, err := json.MarshalIndent(tombstone, "", "  ")
	if err != nil {
		return err
	}
	if int64(len(encoded)+1) > reservation {
		return fmt.Errorf("%w: controlled pair tombstone exceeds its durable reservation", ErrQuota)
	}
	dir, err := s.controlledPairDir(pair.PairID)
	if err != nil {
		return err
	}
	if err := writeJSONAtomic(filepath.Join(dir, controlledPairTombstoneFile), tombstone); err != nil {
		return err
	}
	manifestPath := filepath.Join(dir, controlledPairManifestFile)
	if err := os.Remove(manifestPath); err != nil && !os.IsNotExist(err) {
		return err
	}
	if err := s.pairPersistence.SyncDirectory(dir, "controlled pair tombstone publication"); err != nil {
		return fmt.Errorf("controlled pair tombstone durability is uncertain: %w", err)
	}
	return nil
}
