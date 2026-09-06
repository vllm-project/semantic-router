package evaluationplane

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"strings"
)

const (
	maxExecutionAttestationBytes = int64(64 * 1024 * 1024)
	maxBrokerObservedFieldBytes  = 512
)

var allowedExecutionAttestationHeaders = map[string]bool{
	"x-vsr-selected-model":     true,
	"x-vsr-selected-algorithm": true,
	"x-vsr-selected-recipe":    true,
	"x-vsr-selected-decision":  true,
}

func (s *Store) writeExecutionAttestation(attestation executionAttestation) error {
	if err := validateExecutionAttestationIdentity(attestation.RunID, attestation); err != nil {
		return err
	}
	encoded, err := json.Marshal(attestation)
	if err != nil || int64(len(encoded)) > maxExecutionAttestationBytes {
		return fmt.Errorf("encode evaluation execution attestation")
	}
	path := filepath.Join(s.attestationRoot, attestation.RunID+".json")
	//nolint:gosec // Fixed UUID-derived filename under a validated private directory.
	file, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o600)
	if err != nil {
		return fmt.Errorf("create evaluation execution attestation: %w", err)
	}
	remove := true
	defer func() {
		if remove {
			_ = os.Remove(path)
		}
	}()
	encoded = append(encoded, '\n')
	if _, err = file.Write(encoded); err == nil {
		err = file.Sync()
	}
	closeErr := file.Close()
	if err != nil {
		return fmt.Errorf("write evaluation execution attestation: %w", err)
	}
	if closeErr != nil {
		return fmt.Errorf("close evaluation execution attestation: %w", closeErr)
	}
	if err := syncEvaluationDirectory(s.attestationRoot, "evaluation execution attestations"); err != nil {
		return err
	}
	remove = false
	return nil
}

func (s *Store) readExecutionAttestation(runID string) (executionAttestation, error) {
	if !validClientRequestID(runID) {
		return executionAttestation{}, fmt.Errorf("%w: execution attestation run id is invalid", ErrInvalid)
	}
	path := filepath.Join(s.attestationRoot, runID+".json")
	file, err := openBundleFile(path, os.O_RDONLY)
	if err != nil {
		if os.IsNotExist(err) {
			return executionAttestation{}, fmt.Errorf("%w: execution attestation", ErrNotFound)
		}
		return executionAttestation{}, fmt.Errorf("open evaluation execution attestation: %w", err)
	}
	defer func() { _ = file.Close() }()
	data, err := io.ReadAll(io.LimitReader(file, maxExecutionAttestationBytes+1))
	if err != nil || int64(len(data)) > maxExecutionAttestationBytes {
		return executionAttestation{}, fmt.Errorf("read evaluation execution attestation")
	}
	if err := rejectDuplicateJSONKeys(data); err != nil {
		return executionAttestation{}, fmt.Errorf("decode evaluation execution attestation: %w", err)
	}
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	var attestation executionAttestation
	if err := decoder.Decode(&attestation); err != nil {
		return executionAttestation{}, fmt.Errorf("decode evaluation execution attestation: %w", err)
	}
	if err := ensureJSONEOF(decoder); err != nil {
		return executionAttestation{}, err
	}
	if err := validateExecutionAttestationIdentity(runID, attestation); err != nil {
		return executionAttestation{}, err
	}
	return attestation, nil
}

func (s *Store) rollbackUnanchoredExecutionAttestation(runID string) error {
	runDir, runDirErr := s.checkedRunDir(runID)
	if runDirErr != nil {
		return nil
	}
	if _, err := os.Lstat(filepath.Join(runDir, reportAnchorFileName)); !os.IsNotExist(err) {
		return nil
	}
	removed, err := s.removeExecutionAttestationIfPresent(runID)
	if err != nil {
		return fmt.Errorf("roll back live attestation: %w", err)
	}
	if !removed {
		return nil
	}
	if err := syncEvaluationDirectory(s.attestationRoot, "evaluation execution attestation rollback"); err != nil {
		return fmt.Errorf("sync live attestation rollback: %w", err)
	}
	return nil
}

// readExecutionAttestationForManifest is the durable consumption seam. The
// raw reader above verifies the self-contained file and receipt digests; every
// product read must additionally rebind Router decisions to the immutable run
// manifest so replacing a self-consistent snapshot cannot change its plan.
func (s *Store) readExecutionAttestationForManifest(
	runID string,
	manifest RunManifest,
) (executionAttestation, error) {
	attestation, err := s.readExecutionAttestation(runID)
	if err != nil {
		return executionAttestation{}, err
	}
	if err := validateExecutionAttestationAgainstManifest(attestation, manifest); err != nil {
		return executionAttestation{}, err
	}
	return attestation, nil
}

func (s *Store) readExecutionAttestationForDurableManifest(runID string) (executionAttestation, error) {
	path, err := s.ManifestPath(runID)
	if err != nil {
		return executionAttestation{}, err
	}
	manifest, _, err := readRunManifestStrict(path)
	if err != nil || manifest.RunID != runID {
		return executionAttestation{}, fmt.Errorf("%w: execution attestation manifest is invalid", ErrInvalid)
	}
	return s.readExecutionAttestationForManifest(runID, manifest)
}

func validateExecutionAttestationAgainstManifest(
	attestation executionAttestation,
	manifest RunManifest,
) error {
	if attestation.RunID != manifest.RunID || attestation.ManifestDigest != manifest.ManifestDigest ||
		attestation.TargetID != manifest.Target.ID || attestation.Mode != manifest.Mode ||
		attestation.PolicySnapshotDigest != manifest.PolicySnapshotDigest ||
		attestation.BackendTopologyDigest != manifest.Target.BackendTopologyDigest {
		return fmt.Errorf("%w: execution attestation does not bind the immutable manifest", ErrInvalid)
	}
	for index, entry := range attestation.Entries {
		if entry.Operation == workerBrokerListModels {
			continue
		}
		if err := validateBrokerRoutingRecipeDecision(manifest.Target.Mixture, entry); err != nil {
			return fmt.Errorf("%w: execution attestation entry %d: %w", ErrInvalid, index+1, err)
		}
		if err := validateBrokerMixtureBinding(manifest.Target.Mixture, entry); err != nil {
			return fmt.Errorf("%w: execution attestation entry %d: %w", ErrInvalid, index+1, err)
		}
	}
	return nil
}

func validateExecutionAttestationIdentity(runID string, attestation executionAttestation) error {
	if attestation.SchemaVersion != SchemaVersion ||
		attestation.ContractVersion != executionAttestationContractVersion ||
		attestation.RunID != runID || !validClientRequestID(runID) ||
		!evidenceIDPattern.MatchString(attestation.TargetID) ||
		attestation.Mode != ModeLive || !digestPattern.MatchString(attestation.ManifestDigest) ||
		!digestPattern.MatchString(attestation.PolicySnapshotDigest) ||
		!digestPattern.MatchString(attestation.BackendTopologyDigest) ||
		attestation.StartedAt.IsZero() || attestation.CompletedAt.Before(attestation.StartedAt) ||
		len(attestation.Entries) == 0 || len(attestation.Entries) > maxWorkerBrokerRequests ||
		!digestPattern.MatchString(attestation.Digest) {
		return fmt.Errorf("%w: evaluation execution attestation identity is invalid", ErrInvalid)
	}
	digest, err := executionAttestationDigest(attestation)
	if err != nil || digest != attestation.Digest {
		return fmt.Errorf("%w: evaluation execution attestation digest is invalid", ErrInvalid)
	}
	seenReceipts := make(map[string]bool, len(attestation.Entries))
	modelDiscovery := 0
	expectedRequestID := uint64(1)
	for _, entry := range attestation.Entries {
		if err := validateStoredExecutionAttestationEntry(entry, expectedRequestID); err != nil {
			return err
		}
		if entry.FetchedAt != nil && (entry.FetchedAt.Before(attestation.StartedAt) || entry.FetchedAt.After(attestation.CompletedAt)) {
			return fmt.Errorf("%w: evaluation execution attestation fetched_at lies outside its server window", ErrInvalid)
		}
		expectedRequestID++
		if seenReceipts[entry.BrokerReceipt] {
			return fmt.Errorf("%w: evaluation execution attestation receipt is duplicated", ErrInvalid)
		}
		expected, receiptErr := brokerEntryReceipt(entry)
		if receiptErr != nil || expected != entry.BrokerReceipt {
			return fmt.Errorf("%w: evaluation broker receipt digest is invalid", ErrInvalid)
		}
		seenReceipts[entry.BrokerReceipt] = true
		if entry.Operation == workerBrokerListModels {
			modelDiscovery++
		}
	}
	if modelDiscovery != 1 {
		return fmt.Errorf("%w: evaluation execution attestation requires exactly one model discovery", ErrInvalid)
	}
	return nil
}

func validateStoredExecutionAttestationEntry(entry executionAttestationEntry, expectedID uint64) error {
	if err := validateStoredExecutionAttestationFields(entry, expectedID); err != nil {
		return err
	}
	return validateStoredExecutionAttestationOperation(entry)
}

func validateStoredExecutionAttestationFields(entry executionAttestationEntry, expectedID uint64) error {
	if entry.RequestID != expectedID || !digestPattern.MatchString(entry.RequestDigest) ||
		!digestPattern.MatchString(entry.ResponseDigest) || !digestPattern.MatchString(entry.BrokerReceipt) ||
		entry.LatencyMicroseconds < 0 || entry.Headers == nil ||
		(!entry.UpstreamAttempted && !unattemptedRoutingDecisionUnavailable(entry)) {
		return fmt.Errorf("%w: evaluation execution attestation entry is invalid", ErrInvalid)
	}
	if entry.StatusCode != nil {
		if *entry.StatusCode < 100 || *entry.StatusCode > 599 {
			return fmt.Errorf("%w: evaluation execution attestation status is invalid", ErrInvalid)
		}
	} else if entry.Success {
		return fmt.Errorf("%w: successful broker evidence must include an HTTP status", ErrInvalid)
	}
	if entry.Success && (*entry.StatusCode < 200 || *entry.StatusCode >= 300) {
		return fmt.Errorf("%w: successful broker evidence must have a 2xx status", ErrInvalid)
	}
	for name, value := range entry.Headers {
		if !allowedExecutionAttestationHeaders[name] || value == "" || len(value) > 256 ||
			strings.ContainsAny(value, "\x00\r\n") {
			return fmt.Errorf("%w: evaluation execution attestation header is invalid", ErrInvalid)
		}
	}
	for _, value := range []*string{
		entry.RequestedModel, entry.ArmID, entry.SelectedModel, entry.SelectionStatus, entry.SelectionMethod,
		entry.Recipe, entry.DecisionName, entry.Algorithm,
	} {
		if value != nil && (*value == "" || *value != strings.TrimSpace(*value) ||
			len(*value) > maxBrokerObservedFieldBytes || strings.ContainsAny(*value, "\x00\r\n")) {
			return fmt.Errorf("%w: evaluation execution attestation observation is invalid", ErrInvalid)
		}
	}
	if entry.ResponseContentDigest != nil && !digestPattern.MatchString(*entry.ResponseContentDigest) {
		return fmt.Errorf("%w: evaluation execution attestation response content digest is invalid", ErrInvalid)
	}
	for _, value := range []*int64{entry.InputTokens, entry.OutputTokens} {
		if value != nil && *value < 0 {
			return fmt.Errorf("%w: evaluation execution attestation token count is invalid", ErrInvalid)
		}
	}
	if entry.Quality != nil && (math.IsNaN(*entry.Quality) || math.IsInf(*entry.Quality, 0) ||
		*entry.Quality < 0 || *entry.Quality > 1) {
		return fmt.Errorf("%w: evaluation execution attestation quality is invalid", ErrInvalid)
	}
	if !isMethodLedgerOperation(entry.Operation) && entry.LedgerSealedAt != nil {
		return fmt.Errorf("%w: non-ledger execution attestation claims a ledger seal", ErrInvalid)
	}
	if entry.RoutingRecipeDecision != nil {
		if err := validateRoutingRecipeDecisionSnapshotShape(*entry.RoutingRecipeDecision); err != nil {
			return fmt.Errorf("%w: stored routing recipe decision: %w", ErrInvalid, err)
		}
	}
	return nil
}

func unattemptedRoutingDecisionUnavailable(entry executionAttestationEntry) bool {
	return entry.Operation == workerBrokerRouterEvaluate && !entry.Success && entry.StatusCode == nil &&
		entry.RoutingRecipeDecision != nil && entry.RoutingRecipeDecision.SelectionStatus == "unavailable"
}

func validateStoredExecutionAttestationOperation(entry executionAttestationEntry) error {
	switch entry.Operation {
	case workerBrokerListModels:
		if entry.TrackID != "" || entry.CaseID != "" || entry.AttemptID != "" || !entry.Success ||
			entry.RequestedModel != nil || entry.ArmID != nil || entry.SelectedModel != nil ||
			entry.ResponseContentDigest != nil || entry.Quality != nil ||
			entry.InputTokens != nil || entry.OutputTokens != nil || entry.RoutingRecipeDecision != nil {
			return fmt.Errorf("%w: model discovery attestation is invalid", ErrInvalid)
		}
	case workerBrokerRouterEvaluate:
		if entry.TrackID != "routing" || !evidenceIDPattern.MatchString(entry.CaseID) ||
			!evidenceIDPattern.MatchString(entry.AttemptID) || entry.RequestedModel == nil ||
			entry.FetchedAt == nil || entry.RoutingRecipeDecision == nil || entry.ResponseContentDigest != nil ||
			entry.Recipe == nil {
			return fmt.Errorf("%w: routing broker evidence identity is invalid", ErrInvalid)
		}
	case workerBrokerRoutedChatCompletion:
		if (entry.TrackID != "joint" && entry.TrackID != "multimodal" && entry.TrackID != "capacity") ||
			!evidenceIDPattern.MatchString(entry.CaseID) || !evidenceIDPattern.MatchString(entry.AttemptID) ||
			entry.RequestedModel == nil || (entry.Success &&
			(entry.SelectedModel == nil || entry.ArmID == nil || entry.Recipe == nil || entry.Algorithm == nil ||
				entry.SelectionStatus == nil || entry.SelectionMethod == nil ||
				entry.ResponseContentDigest == nil || entry.InputTokens == nil || entry.OutputTokens == nil)) ||
			(!entry.Success && (entry.SelectionStatus != nil || entry.SelectionMethod != nil || entry.Algorithm != nil)) ||
			entry.RoutingRecipeDecision != nil {
			return fmt.Errorf("%w: routed chat broker evidence identity is invalid", ErrInvalid)
		}
	case workerBrokerArmChatCompletion:
		if entry.TrackID != "model_pool" || !evidenceIDPattern.MatchString(entry.CaseID) ||
			!evidenceIDPattern.MatchString(entry.AttemptID) || entry.RequestedModel == nil || entry.ArmID == nil ||
			(entry.Success && (entry.ResponseContentDigest == nil || entry.InputTokens == nil || entry.OutputTokens == nil)) ||
			entry.RoutingRecipeDecision != nil {
			return fmt.Errorf("%w: arm chat broker evidence identity is invalid", ErrInvalid)
		}
	case workerBrokerAgentTaskLedger:
		if entry.TrackID != "agentic" || !evidenceIDPattern.MatchString(entry.CaseID) ||
			!evidenceIDPattern.MatchString(entry.AttemptID) || entry.FetchedAt == nil || entry.LedgerSealedAt == nil ||
			validateMethodLedgerFreshnessValue(entry) != nil || hasModelExecutionObservation(entry) {
			return fmt.Errorf("%w: agent-task ledger evidence identity is invalid", ErrInvalid)
		}
	case workerBrokerFaultRecoveryLedger:
		if entry.TrackID != "agentic" || !evidenceIDPattern.MatchString(entry.CaseID) ||
			!evidenceIDPattern.MatchString(entry.AttemptID) || entry.FetchedAt == nil || entry.LedgerSealedAt == nil ||
			validateMethodLedgerFreshnessValue(entry) != nil || hasModelExecutionObservation(entry) {
			return fmt.Errorf("%w: fault-recovery ledger evidence identity is invalid", ErrInvalid)
		}
	case workerBrokerHardPolicyLedger:
		if entry.TrackID != "safety" || !evidenceIDPattern.MatchString(entry.CaseID) ||
			!evidenceIDPattern.MatchString(entry.AttemptID) || entry.FetchedAt == nil || entry.LedgerSealedAt == nil ||
			validateMethodLedgerFreshnessValue(entry) != nil || hasModelExecutionObservation(entry) {
			return fmt.Errorf("%w: hard-policy ledger evidence identity is invalid", ErrInvalid)
		}
	case workerBrokerProductionExperimentLedger:
		if entry.TrackID != "preference" || !evidenceIDPattern.MatchString(entry.CaseID) ||
			!evidenceIDPattern.MatchString(entry.AttemptID) || entry.FetchedAt == nil || entry.LedgerSealedAt == nil ||
			validateMethodLedgerFreshnessValue(entry) != nil || hasModelExecutionObservation(entry) {
			return fmt.Errorf("%w: production experiment ledger evidence identity is invalid", ErrInvalid)
		}
	default:
		return fmt.Errorf("%w: evaluation execution attestation operation is invalid", ErrInvalid)
	}
	return nil
}

func validateMethodLedgerFreshnessValue(entry executionAttestationEntry) error {
	if entry.LedgerSealedAt == nil || entry.FetchedAt == nil {
		return fmt.Errorf("method ledger attestation omits its sealed or fetched time")
	}
	return validateMethodLedgerFreshness(*entry.LedgerSealedAt, *entry.FetchedAt)
}

func hasModelExecutionObservation(entry executionAttestationEntry) bool {
	return entry.RequestedModel != nil || entry.ArmID != nil || entry.SelectedModel != nil ||
		entry.ResponseContentDigest != nil || entry.InputTokens != nil || entry.OutputTokens != nil ||
		entry.Quality != nil || entry.RoutingRecipeDecision != nil
}

func (s *Store) removeExecutionAttestationIfPresent(runID string) (bool, error) {
	if !validClientRequestID(runID) {
		return false, fmt.Errorf("%w: execution attestation run id is invalid", ErrInvalid)
	}
	path := filepath.Join(s.attestationRoot, runID+".json")
	info, err := os.Lstat(path)
	if os.IsNotExist(err) {
		return false, nil
	}
	if err != nil {
		return false, fmt.Errorf("inspect evaluation execution attestation: %w", err)
	}
	if !info.Mode().IsRegular() || info.Mode()&os.ModeSymlink != 0 || info.Mode().Perm() != 0o600 {
		return false, fmt.Errorf("evaluation execution attestation is not a private regular file")
	}
	if err := os.Remove(path); err != nil {
		return false, fmt.Errorf("remove evaluation execution attestation: %w", err)
	}
	return true, nil
}

// recoverExecutionAttestationsUnlocked completes the second half of run
// deletion and discards a live transcript that was published before its report
// anchor. Only exact UUID-named private regular files are considered.
func (s *Store) recoverExecutionAttestationsUnlocked() error {
	if err := requirePrivateDirectory(s.attestationRoot); err != nil {
		return fmt.Errorf("validate evaluation execution attestation directory: %w", err)
	}
	entries, err := os.ReadDir(s.attestationRoot)
	if err != nil {
		return fmt.Errorf("list evaluation execution attestations: %w", err)
	}
	removed := false
	for _, entry := range entries {
		name := entry.Name()
		if entry.IsDir() || filepath.Ext(name) != ".json" {
			return fmt.Errorf("evaluation execution attestation directory contains an invalid entry")
		}
		runID := strings.TrimSuffix(name, ".json")
		if !validClientRequestID(runID) {
			return fmt.Errorf("evaluation execution attestation directory contains an invalid entry")
		}
		path := filepath.Join(s.attestationRoot, name)
		info, statErr := os.Lstat(path)
		if statErr != nil || !info.Mode().IsRegular() || info.Mode()&os.ModeSymlink != 0 || info.Mode().Perm() != 0o600 {
			return fmt.Errorf("evaluation execution attestation %s is not a private regular file", runID)
		}

		keep := false
		if runInfo, runErr := os.Lstat(filepath.Join(s.runsRoot, runID)); runErr == nil &&
			runInfo.IsDir() && runInfo.Mode()&os.ModeSymlink == 0 && runInfo.Mode().Perm() == 0o700 {
			anchor, anchorErr := s.readReportAnchor(runID)
			attestation, attestationErr := s.readExecutionAttestationForDurableManifest(runID)
			keep = anchorErr == nil && attestationErr == nil &&
				anchor.ExecutionAttestationDigest != "" &&
				anchor.ExecutionAttestationDigest == attestation.Digest
		}
		if keep {
			continue
		}
		deleted, removeErr := s.removeExecutionAttestationIfPresent(runID)
		if removeErr != nil {
			return removeErr
		}
		removed = removed || deleted
	}
	if removed {
		return syncEvaluationDirectory(s.attestationRoot, "evaluation execution attestation recovery")
	}
	return nil
}
