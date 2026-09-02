package evaluationplane

import (
	"encoding/json"
	"fmt"
)

// withEvidenceSerialization scopes evidence publication to one durable root.
// Different roots remain independent while every publisher for the same root
// observes one transaction boundary. Transactions using this evidence-only
// seam must never acquire lifecycle.mu; lifecycle-bound work enters through
// withEvidencePublication so the global lifecycle -> evidence order is fixed.
func (s *Store) withEvidenceSerialization(transaction func() error) error {
	s.lifecycle.evidenceMu.Lock()
	defer s.lifecycle.evidenceMu.Unlock()
	return transaction()
}

// withEvidencePublication owns the complete lock order for a lifecycle-bound
// evidence transaction. Callers inside the transaction must use the
// *DuringPublication helpers rather than attempting to acquire either outer
// lock again.
func (s *Store) withEvidencePublication(transaction func() error) error {
	s.lifecycle.mu.Lock()
	defer s.lifecycle.mu.Unlock()
	return s.withEvidenceSerialization(transaction)
}

// writeLifecycleBoundExecutionAttestationDuringPublication is the non-locking
// attestation writer. The caller owns the lifecycle and evidence-publication
// locks through withEvidencePublication.
func (s *Store) writeLifecycleBoundExecutionAttestationDuringPublication(
	attestation executionAttestation,
	manifest RunManifest,
) error {
	if err := validateExecutionAttestationAgainstManifest(attestation, manifest); err != nil {
		return err
	}
	encoded, err := json.Marshal(attestation)
	if err != nil || int64(len(encoded)) > maxExecutionAttestationBytes {
		return fmt.Errorf("encode evaluation execution attestation")
	}
	publicationBytes := int64(len(encoded) + 1)

	s.runIndex.coordinator.Lock()
	defer s.runIndex.coordinator.Unlock()
	s.mu.Lock()
	defer s.mu.Unlock()
	if err := s.requireEvidenceQuotaUnlocked(
		attestation.RunID, publicationBytes, 0, publicationBytes,
	); err != nil {
		return err
	}
	return s.writeExecutionAttestation(attestation)
}

// importWorkerEvidence publishes worker artifacts under the same lifecycle,
// quota, index, and evidence lock order used by every other durable mutation.
func (s *Store) importWorkerEvidence(staging *workerStaging) error {
	return s.withEvidencePublication(func() error {
		s.runIndex.coordinator.Lock()
		defer s.runIndex.coordinator.Unlock()
		s.mu.Lock()
		defer s.mu.Unlock()
		return staging.importEvidenceDuringPublication(maxWorkerBundleBytes, s.requireEvidenceQuotaUnlocked)
	})
}
