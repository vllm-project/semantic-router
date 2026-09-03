package recipe

import (
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestFinalizationCleanupFailureRetainsRetryableJournal(t *testing.T) {
	store := NewStore(StoreOptions{Root: filepath.Join(t.TempDir(), "store"), ConfigPath: filepath.Join(t.TempDir(), "config.yaml")})
	target := "sha256:" + strings.Repeat("a", 64)
	transaction, err := store.BeginActivation(target, []byte("previous"))
	if err != nil {
		t.Fatal(err)
	}
	transaction.State = "committing"
	transaction.CommitConfigDigest = "sha256:" + strings.Repeat("b", 64)
	if writeErr := writeJSONAtomically(store.transactionPath(), transaction, 0o600); writeErr != nil {
		t.Fatal(writeErr)
	}
	transactionDir := filepath.Join(store.root, "transactions", transaction.ID)
	if writeErr := os.WriteFile(filepath.Join(transactionDir, "orphan"), []byte("orphan"), 0o600); writeErr != nil {
		t.Fatal(writeErr)
	}
	if finalizationErr := store.finalizeTransactionLocked(transaction); finalizationErr == nil {
		t.Fatal("finalizeTransactionLocked() unexpectedly hid cleanup failure")
	}
	current, err := store.readTransaction()
	if err != nil || current.State != ActivationFinalizing {
		t.Fatalf("retry journal = %+v, %v", current, err)
	}
	if err := os.Remove(filepath.Join(transactionDir, "orphan")); err != nil {
		t.Fatal(err)
	}
	if err := store.finalizeTransactionLocked(current); err != nil {
		t.Fatalf("retry finalization: %v", err)
	}
	if _, err := os.Lstat(store.transactionPath()); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("journal still exists after retry: %v", err)
	}
}

func TestRollbackCleanupFailureRetainsRetryableJournal(t *testing.T) {
	store := NewStore(StoreOptions{Root: filepath.Join(t.TempDir(), "store"), ConfigPath: filepath.Join(t.TempDir(), "config.yaml")})
	target := "sha256:" + strings.Repeat("c", 64)
	transaction, err := store.BeginActivation(target, []byte("previous"))
	if err != nil {
		t.Fatal(err)
	}
	transactionDir := filepath.Join(store.root, "transactions", transaction.ID)
	orphan := filepath.Join(transactionDir, "orphan")
	if writeErr := os.WriteFile(orphan, []byte("orphan"), 0o600); writeErr != nil {
		t.Fatal(writeErr)
	}
	if rollbackErr := store.CompleteRollback(transaction); rollbackErr == nil {
		t.Fatal("CompleteRollback() unexpectedly hid cleanup failure")
	}
	current, _, err := store.Transaction()
	if err != nil || current.State != ActivationRollbackFinalizing {
		t.Fatalf("rollback retry journal = %+v, %v", current, err)
	}
	if markerErr := store.MarkTransactionInconsistent(transaction); markerErr != nil {
		t.Fatal(markerErr)
	}
	current, _, err = store.Transaction()
	if err != nil || current.State != ActivationRollbackFinalizing {
		t.Fatalf("rollback marker was overwritten = %+v, %v", current, err)
	}
	if err := os.Remove(orphan); err != nil {
		t.Fatal(err)
	}
	if err := store.CompleteRollback(current); err != nil {
		t.Fatalf("retry rollback finalization: %v", err)
	}
	if _, err := os.Lstat(store.transactionPath()); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("rollback journal still exists after retry: %v", err)
	}
}

func TestSourceBaselineRefreshesAtomicallyByDigest(t *testing.T) {
	store := NewStore(StoreOptions{Root: filepath.Join(t.TempDir(), "store"), ConfigPath: filepath.Join(t.TempDir(), "config.yaml")})
	first := []byte("version: v0.3\n# source\n")
	if err := store.RefreshSourceBaseline(first); err != nil {
		t.Fatal(err)
	}
	second := []byte("version: v0.3\n# edited source\n")
	if err := store.RefreshSourceBaseline(second); err != nil {
		t.Fatal(err)
	}
	got, err := store.SourceBaseline()
	if err != nil || string(got) != string(second) {
		t.Fatalf("SourceBaseline() = %q, %v", got, err)
	}
	if err := os.WriteFile(store.sourceBaselineObjectPath(digestBytes(second)), []byte("tampered"), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := store.SourceBaseline(); err == nil {
		t.Fatal("tampered source baseline was accepted")
	}
}

func TestUnpublishedBaselineObjectDoesNotReplaceCurrentPointer(t *testing.T) {
	store := NewStore(StoreOptions{Root: filepath.Join(t.TempDir(), "store"), ConfigPath: filepath.Join(t.TempDir(), "config.yaml")})
	current := []byte("version: v0.3\n# current\n")
	if err := store.RefreshSourceBaseline(current); err != nil {
		t.Fatal(err)
	}
	unpublished := []byte("version: v0.3\n# interrupted refresh\n")
	objectPath := store.sourceBaselineObjectPath(digestBytes(unpublished))
	if err := writeFileAtomically(objectPath, unpublished, 0o600); err != nil {
		t.Fatal(err)
	}
	got, err := store.SourceBaseline()
	if err != nil || string(got) != string(current) {
		t.Fatalf("SourceBaseline() = %q, %v", got, err)
	}
}
