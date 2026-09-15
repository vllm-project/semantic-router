package evaluationplane

import (
	"testing"
	"time"
)

func TestEvidencePublicationTransactionAcceptsLiveAttestation(t *testing.T) {
	store := newPrivateTestStore(t)
	runID := newTestClientRequestID()
	makePrivateRunDirectory(t, store, runID)
	attestation := validExecutionAttestation(t, runID)

	done := make(chan error, 1)
	go func() {
		done <- store.withEvidencePublication(func() error {
			return store.writeLifecycleBoundExecutionAttestationDuringPublication(attestation, RunManifest{
				RunID: attestation.RunID, ManifestDigest: attestation.ManifestDigest, Mode: attestation.Mode,
				PolicySnapshotDigest: attestation.PolicySnapshotDigest,
				Target:               ManifestTarget{ID: attestation.TargetID, BackendTopologyDigest: attestation.BackendTopologyDigest},
			})
		})
	}()

	select {
	case err := <-done:
		if err != nil {
			t.Fatalf("publish live attestation in one evidence transaction: %v", err)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("live attestation publication deadlocked on the transaction lock")
	}
	if _, err := store.readExecutionAttestation(runID); err != nil {
		t.Fatalf("read transactionally published live attestation: %v", err)
	}
}
