package recipe

import (
	"encoding/hex"
	"errors"
	"net/http"
	"os"
	"path/filepath"
	"strings"
)

func (s *Store) ActivationTarget(recipeDigest string, acknowledgeWarnings bool) (PackageSummary, []byte, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if err := s.ensureLayout(); err != nil {
		return PackageSummary{}, nil, err
	}
	record, err := s.findRecordByDigest(recipeDigest)
	if err != nil {
		return PackageSummary{}, nil, err
	}
	if len(record.Warnings) > 0 && !acknowledgeWarnings {
		return PackageSummary{}, nil, wrapPackageError(ErrorWarningsNotAcknowledged, http.StatusConflict, "Recipe package warnings must be acknowledged before activation.", nil)
	}
	directory, _ := s.ObjectDirectory(record.RecipeDigest)
	files, err := readRecipeObject(directory)
	if err != nil || digestRecipe(files) != record.RecipeDigest || digestBytes(files["config"]) != record.ConfigDigest {
		return PackageSummary{}, nil, wrapPackageError(ErrorInvalidRecipe, http.StatusUnprocessableEntity, "Installed Recipe package content is inconsistent.", err)
	}
	active, state, _ := s.activationStatusLocked()
	return summaryFromRecord(record, activeDigestForState(active, state)), files["config"], nil
}

func (s *Store) BeginActivation(targetRecipeDigest string, previousConfig []byte) (ActivationTransaction, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.beginActivationLocked(ActivationOperationActivate, targetRecipeDigest, previousConfig)
}

func (s *Store) BeginDeactivation(previousConfig []byte) (ActivationTransaction, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	active, err := s.readActivePointer()
	if err != nil {
		return ActivationTransaction{}, err
	}
	return s.beginActivationLocked(ActivationOperationDeactivate, active.RecipeDigest, previousConfig)
}

func (s *Store) beginActivationLocked(operation, targetRecipeDigest string, previousConfig []byte) (ActivationTransaction, error) {
	if err := s.ensureLayout(); err != nil {
		return ActivationTransaction{}, err
	}
	if _, err := s.readTransaction(); err == nil {
		return ActivationTransaction{}, wrapPackageError(ErrorActivationConflict, http.StatusConflict, "A Recipe activation recovery is already in progress.", nil)
	} else if !errors.Is(err, os.ErrNotExist) {
		return ActivationTransaction{}, err
	}
	target, err := normalizeRecipeDigest(targetRecipeDigest)
	if err != nil {
		return ActivationTransaction{}, err
	}
	id, err := randomID()
	if err != nil {
		return ActivationTransaction{}, err
	}
	transactionDirectory := filepath.Join(s.root, "transactions", id)
	if err := os.Mkdir(transactionDirectory, 0o700); err != nil {
		return ActivationTransaction{}, err
	}
	backupRelative := filepath.Join("transactions", id, "previous-config.yaml")
	backupPath := filepath.Join(s.root, backupRelative)
	if err := writeFileAtomically(backupPath, previousConfig, 0o600); err != nil {
		_ = os.RemoveAll(transactionDirectory)
		return ActivationTransaction{}, err
	}
	if err := syncDirectory(filepath.Join(s.root, "transactions")); err != nil {
		_ = os.RemoveAll(transactionDirectory)
		return ActivationTransaction{}, err
	}
	active, activeErr := s.readActivePointer()
	var previousPointer *ActivePointer
	if activeErr == nil {
		previousPointer = &active
	} else if !errors.Is(activeErr, os.ErrNotExist) {
		_ = os.RemoveAll(transactionDirectory)
		return ActivationTransaction{}, activeErr
	}
	transaction := ActivationTransaction{
		SchemaVersion:        activationTxnSchema,
		State:                "pending",
		Operation:            operation,
		TopologyMode:         ActivationTopologyNone,
		ID:                   id,
		TargetRecipeDigest:   target,
		PreviousRecipeDigest: active.RecipeDigest,
		PreviousPointer:      previousPointer,
		PreviousConfigDigest: digestBytes(previousConfig),
		PreviousConfigBackup: filepath.ToSlash(backupRelative),
		StartedAt:            s.now().UTC(),
	}
	if err := writeJSONAtomically(s.transactionPath(), transaction, 0o600); err != nil {
		_ = os.RemoveAll(transactionDirectory)
		return ActivationTransaction{}, err
	}
	return transaction, nil
}

func (s *Store) Transaction() (ActivationTransaction, []byte, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	transaction, err := s.readTransaction()
	if err != nil {
		return ActivationTransaction{}, nil, err
	}
	if transaction.State == ActivationFinalizing || transaction.State == ActivationRollbackFinalizing {
		return transaction, nil, nil
	}
	previous, err := s.readTransactionBackup(transaction)
	if err != nil {
		return transaction, nil, err
	}
	return transaction, previous, nil
}

func (s *Store) CommitActivation(transaction ActivationTransaction, pointer ActivePointer) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	current, err := s.readTransaction()
	if err != nil || current.ID != transaction.ID || current.State != "pending" || current.Operation != ActivationOperationActivate {
		return wrapPackageError(ErrorActivationConflict, http.StatusConflict, "Recipe activation transaction changed unexpectedly.", err)
	}
	pointer.SchemaVersion = activePointerSchema
	pointer.ActivatedAt = s.now().UTC()
	if err := s.validatePointerFields(pointer, transaction.TargetRecipeDigest); err != nil {
		return err
	}
	if err := writeJSONAtomically(filepath.Join(s.root, "active.json"), pointer, 0o600); err != nil {
		return err
	}
	current.State = "committing"
	current.CommitConfigDigest = pointer.RealizedConfigDigest
	if err := writeJSONAtomically(s.transactionPath(), current, 0o600); err != nil {
		return err
	}
	return s.finalizeTransactionLocked(current)
}

// PrepareActivationCommit durably publishes the verified active pointer while
// retaining the transaction and topology journal. A topology-aware caller must
// finish its idempotent container cleanup before FinalizeActivationCommit.
func (s *Store) PrepareActivationCommit(transaction ActivationTransaction, pointer ActivePointer) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.prepareActivationCommitLocked(transaction, pointer)
}

func (s *Store) prepareActivationCommitLocked(transaction ActivationTransaction, pointer ActivePointer) error {
	current, err := s.readTransaction()
	if err != nil || current.ID != transaction.ID || current.State != "pending" || current.Operation != ActivationOperationActivate {
		return wrapPackageError(ErrorActivationConflict, http.StatusConflict, "Recipe activation transaction changed unexpectedly.", err)
	}
	pointer.SchemaVersion = activePointerSchema
	pointer.ActivatedAt = s.now().UTC()
	if err := s.validatePointerFields(pointer, transaction.TargetRecipeDigest); err != nil {
		return err
	}
	if err := writeJSONAtomically(filepath.Join(s.root, "active.json"), pointer, 0o600); err != nil {
		return err
	}
	current.State = "committing"
	current.CommitConfigDigest = pointer.RealizedConfigDigest
	return writeJSONAtomically(s.transactionPath(), current, 0o600)
}

func (s *Store) FinalizeActivationCommit(transaction ActivationTransaction) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	current, err := s.readTransaction()
	if err != nil || current.ID != transaction.ID || (current.State != "committing" && current.State != ActivationFinalizing) || current.Operation != ActivationOperationActivate {
		return wrapPackageError(ErrorActivationConflict, http.StatusConflict, "Recipe activation commit changed unexpectedly.", err)
	}
	active, err := s.readActivePointer()
	if err != nil || active.RecipeDigest != current.TargetRecipeDigest {
		return wrapPackageError(ErrorActivationConflict, http.StatusConflict, "Recipe activation pointer changed unexpectedly.", err)
	}
	return s.finalizeTransactionLocked(current)
}

func (s *Store) CommitDeactivation(transaction ActivationTransaction) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	current, err := s.readTransaction()
	if err != nil || current.ID != transaction.ID || current.State != "pending" || current.Operation != ActivationOperationDeactivate {
		return wrapPackageError(ErrorActivationConflict, http.StatusConflict, "Recipe deactivation transaction changed unexpectedly.", err)
	}
	active, err := s.readActivePointer()
	if err != nil || active.RecipeDigest != current.TargetRecipeDigest {
		return wrapPackageError(ErrorActivationConflict, http.StatusConflict, "Active Recipe changed during deactivation.", err)
	}
	if removeErr := os.Remove(filepath.Join(s.root, "active.json")); removeErr != nil {
		return removeErr
	}
	if syncErr := syncDirectory(s.root); syncErr != nil {
		return syncErr
	}
	config, err := readBoundedFile(s.configPath, maxConfigBytes)
	if err != nil {
		return err
	}
	current.State = "committing"
	current.CommitConfigDigest = digestBytes(config)
	if err := writeJSONAtomically(s.transactionPath(), current, 0o600); err != nil {
		return err
	}
	return s.finalizeTransactionLocked(current)
}

func (s *Store) PrepareDeactivationCommit(transaction ActivationTransaction) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.prepareDeactivationCommitLocked(transaction)
}

func (s *Store) prepareDeactivationCommitLocked(transaction ActivationTransaction) error {
	current, err := s.readTransaction()
	if err != nil || current.ID != transaction.ID || current.State != "pending" || current.Operation != ActivationOperationDeactivate {
		return wrapPackageError(ErrorActivationConflict, http.StatusConflict, "Recipe deactivation transaction changed unexpectedly.", err)
	}
	active, err := s.readActivePointer()
	if err != nil || active.RecipeDigest != current.TargetRecipeDigest {
		return wrapPackageError(ErrorActivationConflict, http.StatusConflict, "Active Recipe changed during deactivation.", err)
	}
	if removeErr := os.Remove(filepath.Join(s.root, "active.json")); removeErr != nil {
		return removeErr
	}
	// The pointer removal must be durable before the journal commit. Until the
	// journal is removed, recovery can still restore the prior package pointer.
	if syncErr := syncDirectory(s.root); syncErr != nil {
		return syncErr
	}
	config, err := readBoundedFile(s.configPath, maxConfigBytes)
	if err != nil {
		return err
	}
	current.State = "committing"
	current.CommitConfigDigest = digestBytes(config)
	return writeJSONAtomically(s.transactionPath(), current, 0o600)
}

func (s *Store) FinalizeDeactivationCommit(transaction ActivationTransaction) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	current, err := s.readTransaction()
	if err != nil || current.ID != transaction.ID || (current.State != "committing" && current.State != ActivationFinalizing) || current.Operation != ActivationOperationDeactivate {
		return wrapPackageError(ErrorActivationConflict, http.StatusConflict, "Recipe deactivation commit changed unexpectedly.", err)
	}
	if _, activeErr := s.readActivePointer(); !errors.Is(activeErr, os.ErrNotExist) {
		return wrapPackageError(ErrorActivationConflict, http.StatusConflict, "Recipe deactivation pointer changed unexpectedly.", activeErr)
	}
	return s.finalizeTransactionLocked(current)
}

func (s *Store) CompleteRollback(transaction ActivationTransaction) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	current, err := s.readTransaction()
	if err != nil {
		return err
	}
	if current.ID != transaction.ID {
		return errors.New("activation transaction changed unexpectedly")
	}
	if current.State != ActivationRollbackFinalizing {
		if current.State != "pending" && current.State != ActivationInconsistent {
			return errors.New("activation transaction is not ready for rollback finalization")
		}
		if err := s.restorePreviousPointer(current); err != nil {
			return err
		}
		current.State = ActivationRollbackFinalizing
		if err := writeJSONAtomically(s.transactionPath(), current, 0o600); err != nil {
			return err
		}
	}
	return s.finishTransactionCleanupLocked(current)
}

func (s *Store) MarkTransactionInconsistent(transaction ActivationTransaction) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	current, err := s.readTransaction()
	if err != nil {
		return err
	}
	if current.ID != transaction.ID {
		return errors.New("activation transaction changed unexpectedly")
	}
	if current.State == ActivationFinalizing || current.State == ActivationRollbackFinalizing {
		return nil
	}
	current.State = ActivationInconsistent
	return writeJSONAtomically(s.transactionPath(), current, 0o600)
}

func (s *Store) readTransaction() (ActivationTransaction, error) {
	var transaction ActivationTransaction
	if err := readStrictJSONFile(s.transactionPath(), 128<<10, &transaction); err != nil {
		return ActivationTransaction{}, err
	}
	if err := validateActivationTransactionEnvelope(transaction); err != nil {
		return ActivationTransaction{}, err
	}
	normalizeLegacyActivationTransaction(&transaction)
	if err := validateActivationTransactionIdentity(transaction); err != nil {
		return ActivationTransaction{}, err
	}
	if err := validatePreviousActivePointer(transaction); err != nil {
		return ActivationTransaction{}, err
	}
	if err := validateCommitConfigDigest(transaction); err != nil {
		return ActivationTransaction{}, err
	}
	return transaction, nil
}

func validateActivationTransactionEnvelope(transaction ActivationTransaction) error {
	if transaction.SchemaVersion != activationTxnSchema || (transaction.State != "pending" && transaction.State != "committing" && transaction.State != ActivationFinalizing && transaction.State != ActivationRollbackFinalizing && transaction.State != ActivationInconsistent) {
		return errors.New("invalid activation transaction")
	}
	return nil
}

func normalizeLegacyActivationTransaction(transaction *ActivationTransaction) {
	// Journals produced before managed topology support used the same v1 schema
	// without operation/topology fields. They can only represent a pending or
	// inconsistent activation and are safely normalized to the old hot-switch
	// rollback behavior. New commit/finalization states never use this fallback.
	if transaction.Operation == "" && transaction.TopologyMode == "" &&
		(transaction.State == "pending" || transaction.State == ActivationInconsistent) {
		transaction.Operation = ActivationOperationActivate
		transaction.TopologyMode = ActivationTopologyNone
	}
}

func validateActivationTransactionIdentity(transaction ActivationTransaction) error {
	if transaction.Operation != ActivationOperationActivate && transaction.Operation != ActivationOperationDeactivate {
		return errors.New("invalid activation transaction operation")
	}
	if transaction.TopologyMode != ActivationTopologyNone && transaction.TopologyMode != ActivationTopologyManaged {
		return errors.New("invalid activation transaction topology mode")
	}
	if !validTransactionID(transaction.ID) || !validDigest(transaction.TargetRecipeDigest) || !validDigest(transaction.PreviousConfigDigest) {
		return errors.New("invalid activation transaction identity")
	}
	if transaction.PreviousRecipeDigest != "" && !validDigest(transaction.PreviousRecipeDigest) {
		return errors.New("invalid previous Recipe digest")
	}
	return nil
}

func validatePreviousActivePointer(transaction ActivationTransaction) error {
	if transaction.PreviousPointer == nil {
		return nil
	}
	if transaction.PreviousPointer.RecipeDigest != transaction.PreviousRecipeDigest ||
		transaction.PreviousPointer.SchemaVersion != activePointerSchema ||
		!validDigest(transaction.PreviousPointer.ConfigDigest) ||
		!validDigest(transaction.PreviousPointer.RealizedConfigDigest) {
		return errors.New("invalid previous active pointer")
	}
	return nil
}

func validateCommitConfigDigest(transaction ActivationTransaction) error {
	if transaction.State == "committing" || transaction.State == ActivationFinalizing {
		if !validDigest(transaction.CommitConfigDigest) {
			return errors.New("invalid committing config digest")
		}
	} else if transaction.CommitConfigDigest != "" {
		return errors.New("unexpected commit config digest")
	}
	return nil
}

func validTransactionID(value string) bool {
	if len(value) != 32 || value != strings.ToLower(value) {
		return false
	}
	decoded, err := hex.DecodeString(value)
	return err == nil && len(decoded) == 16
}

func (s *Store) readTransactionBackup(transaction ActivationTransaction) ([]byte, error) {
	relative := filepath.Clean(filepath.FromSlash(transaction.PreviousConfigBackup))
	expected := filepath.Join("transactions", transaction.ID, "previous-config.yaml")
	if relative != expected || filepath.IsAbs(relative) || strings.HasPrefix(relative, "..") {
		return nil, errors.New("invalid activation backup path")
	}
	transactionDirectory := filepath.Join(s.root, "transactions", transaction.ID)
	info, err := os.Lstat(transactionDirectory)
	if err != nil || info.Mode()&os.ModeSymlink != 0 || !info.IsDir() {
		return nil, errors.New("activation backup directory is invalid")
	}
	data, err := readBoundedFile(filepath.Join(s.root, relative), maxConfigBytes)
	if err != nil {
		return nil, err
	}
	if digestBytes(data) != transaction.PreviousConfigDigest {
		return nil, errors.New("activation backup digest mismatch")
	}
	return data, nil
}

func (s *Store) validatePointerFields(pointer ActivePointer, target string) error {
	if pointer.RecipeDigest != target || !validDigest(pointer.RecipeDigest) || !validDigest(pointer.ConfigDigest) || !validDigest(pointer.RealizedConfigDigest) {
		return errors.New("invalid active pointer fields")
	}
	return nil
}

// finalizeTransactionLocked first persists a finalizing marker, then removes
// transaction artifacts, and unlinks the durable journal last. Any cleanup or
// durability error remains visible and can be retried from that marker without
// replaying topology changes or rolling back an already verified runtime.
func (s *Store) finalizeTransactionLocked(transaction ActivationTransaction) error {
	if transaction.State != ActivationFinalizing {
		if transaction.State != "committing" || !validDigest(transaction.CommitConfigDigest) {
			return errors.New("activation transaction is not ready for finalization")
		}
		transaction.State = ActivationFinalizing
		if err := writeJSONAtomically(s.transactionPath(), transaction, 0o600); err != nil {
			return err
		}
	}
	return s.finishTransactionCleanupLocked(transaction)
}

func (s *Store) finishTransactionCleanupLocked(transaction ActivationTransaction) error {
	if err := s.cleanupTransactionDirectory(transaction); err != nil {
		return err
	}
	if err := os.Remove(s.transactionPath()); err != nil && !errors.Is(err, os.ErrNotExist) {
		return err
	}
	return syncDirectory(s.root)
}

func (s *Store) cleanupTransactionDirectory(transaction ActivationTransaction) error {
	transactionsRoot := filepath.Join(s.root, "transactions")
	if err := os.Remove(filepath.Join(transactionsRoot, transaction.ID, "topology.json")); err != nil && !errors.Is(err, os.ErrNotExist) {
		return err
	}
	if err := os.Remove(filepath.Join(transactionsRoot, transaction.ID, "previous-config.yaml")); err != nil && !errors.Is(err, os.ErrNotExist) {
		return err
	}
	if err := os.Remove(filepath.Join(transactionsRoot, transaction.ID)); err != nil && !errors.Is(err, os.ErrNotExist) {
		return err
	}
	return syncDirectory(transactionsRoot)
}

func (s *Store) restorePreviousPointer(transaction ActivationTransaction) error {
	path := filepath.Join(s.root, "active.json")
	if transaction.PreviousPointer == nil {
		if err := os.Remove(path); err != nil && !errors.Is(err, os.ErrNotExist) {
			return err
		}
		return syncDirectory(s.root)
	}
	return writeJSONAtomically(path, transaction.PreviousPointer, 0o600)
}

func (s *Store) transactionPath() string {
	return filepath.Join(s.root, "activation-pending.json")
}
