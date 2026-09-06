package evaluationplane

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
)

func (s *Store) lifecycleCollectionPath() string {
	return filepath.Join(s.collectionRoot, lifecycleCollectionFileName)
}

func (s *Store) readLifecycleCollectionTransactionUnlocked() (lifecycleCollectionTransaction, bool, error) {
	path := s.lifecycleCollectionPath()
	if _, err := os.Lstat(path); os.IsNotExist(err) {
		return lifecycleCollectionTransaction{}, false, nil
	} else if err != nil {
		return lifecycleCollectionTransaction{}, false, fmt.Errorf("inspect lifecycle collection transaction: %w", err)
	}
	data, err := readEvidenceBytes(path, maxStructuredArtifactBytes)
	if err != nil {
		return lifecycleCollectionTransaction{}, false, err
	}
	if len(data) == 0 {
		return lifecycleCollectionTransaction{}, false, fmt.Errorf("%w: lifecycle collection log is empty", ErrInvalid)
	}
	if data[len(data)-1] != '\n' {
		lastComplete := bytes.LastIndexByte(data, '\n')
		if lastComplete < 0 {
			return lifecycleCollectionTransaction{}, false, fmt.Errorf(
				"%w: lifecycle collection header is incomplete", ErrInvalid,
			)
		}
		// A short append cannot be trusted as progress, but the immutable
		// header and every newline-terminated record before it remain readable.
		// Only startup authority or the matching retry may truncate this tail.
		data = data[:lastComplete+1]
	}
	lines := bytes.Split(bytes.TrimSuffix(data, []byte{'\n'}), []byte{'\n'})
	if len(lines) == 0 || len(lines) > maxLifecycleCollectionCandidates+1 {
		return lifecycleCollectionTransaction{}, false, fmt.Errorf("%w: lifecycle collection log is invalid", ErrInvalid)
	}
	var header lifecycleCollectionHeader
	if err := decodeLifecycleCollectionRecord(lines[0], &header); err != nil {
		return lifecycleCollectionTransaction{}, false, err
	}
	if err := validateLifecycleCollectionHeader(header); err != nil {
		return lifecycleCollectionTransaction{}, false, err
	}
	transaction := lifecycleCollectionTransaction{
		SchemaVersion:        header.SchemaVersion,
		State:                header.State,
		ActorPrincipalDigest: header.ActorPrincipalDigest,
		Plan:                 header.Plan,
		PlanIdentity:         header.PlanIdentity,
		Result:               collectionResultThrough(header.Plan, 0),
		ReceiptDigest:        header.HeaderDigest,
	}
	for index, line := range lines[1:] {
		if len(line)+1 > maxLifecycleCollectionProgressBytes {
			return lifecycleCollectionTransaction{}, false, fmt.Errorf("%w: lifecycle collection progress is oversized", ErrInvalid)
		}
		var progress lifecycleCollectionProgress
		if err := decodeLifecycleCollectionRecord(line, &progress); err != nil {
			return lifecycleCollectionTransaction{}, false, err
		}
		if err := validateLifecycleCollectionProgress(progress, transaction, index+1); err != nil {
			return lifecycleCollectionTransaction{}, false, err
		}
		transaction.State = progress.State
		transaction.Next = progress.Next
		transaction.Result = collectionResultThrough(transaction.Plan, transaction.Next)
		transaction.ReceiptDigest = progress.ProgressDigest
	}
	if err := validateLifecycleCollectionTransaction(transaction); err != nil {
		return lifecycleCollectionTransaction{}, false, err
	}
	return transaction, true, nil
}

func (s *Store) resolveLifecycleCollectionTransactionUnlocked() (lifecycleCollectionTransaction, error) {
	if err := s.collectionPersistence.Resolve(s.lifecycleCollectionPath(), s.collectionRoot); err != nil {
		return lifecycleCollectionTransaction{}, fmt.Errorf(
			"%w: lifecycle collection transaction durability is uncertain: %w", ErrConflict, err,
		)
	}
	transaction, exists, err := s.readLifecycleCollectionTransactionUnlocked()
	if err != nil {
		return lifecycleCollectionTransaction{}, err
	}
	if !exists {
		return lifecycleCollectionTransaction{}, fmt.Errorf("%w: lifecycle collection transaction disappeared", ErrConflict)
	}
	return transaction, nil
}

func (s *Store) persistLifecycleCollectionTransactionUnlocked(
	transaction *lifecycleCollectionTransaction,
) error {
	s.lifecycle.pendingCollection = &pendingLifecycleCollectionProjection{
		ActorPrincipalDigest: transaction.ActorPrincipalDigest,
		PlanDigest:           transaction.Plan.PlanDigest,
		Transaction:          *transaction,
	}
	var err error
	if transaction.ReceiptDigest == "" {
		err = s.persistLifecycleCollectionHeaderUnlocked(transaction)
	} else {
		err = s.persistLifecycleCollectionProgressUnlocked(transaction)
	}
	if err != nil {
		return fmt.Errorf(
			"%w: lifecycle collection transaction durability is uncertain: %w", ErrConflict, err,
		)
	}
	s.clearPendingLifecycleCollectionUnlocked(
		transaction.ActorPrincipalDigest,
		transaction.Plan.PlanDigest,
	)
	return nil
}

func (s *Store) persistLifecycleCollectionHeaderUnlocked(
	transaction *lifecycleCollectionTransaction,
) error {
	header := lifecycleCollectionHeader{
		RecordType: "plan", SchemaVersion: transaction.SchemaVersion,
		ActorPrincipalDigest: transaction.ActorPrincipalDigest,
		State:                transaction.State, Plan: transaction.Plan, PlanIdentity: transaction.PlanIdentity,
	}
	header.HeaderDigest = lifecycleCollectionHeaderDigest(header)
	if err := validateLifecycleCollectionHeader(header); err != nil {
		return err
	}
	if err := s.collectionPersistence.WriteHeader(s.lifecycleCollectionPath(), header); err != nil {
		return err
	}
	transaction.ReceiptDigest = header.HeaderDigest
	return nil
}

func (s *Store) persistLifecycleCollectionProgressUnlocked(
	transaction *lifecycleCollectionTransaction,
) error {
	progress := lifecycleCollectionProgress{
		RecordType: "progress", SchemaVersion: lifecycleCollectionSchemaVersion,
		ActorPrincipalDigest: transaction.ActorPrincipalDigest,
		PlanDigest:           transaction.Plan.PlanDigest, State: transaction.State, Next: transaction.Next,
		PreviousDigest: transaction.ReceiptDigest,
	}
	progress.ProgressDigest = lifecycleCollectionProgressDigest(progress)
	encoded, err := json.Marshal(progress)
	if err != nil {
		return err
	}
	if len(encoded)+1 > maxLifecycleCollectionProgressBytes {
		return fmt.Errorf("lifecycle collection progress record exceeds its bound")
	}
	if err := s.collectionPersistence.AppendProgress(s.lifecycleCollectionPath(), progress); err != nil {
		return err
	}
	transaction.ReceiptDigest = progress.ProgressDigest
	return nil
}

func (s *Store) clearPendingLifecycleCollectionUnlocked(actorPrincipalDigest, planDigest string) {
	pending := s.lifecycle.pendingCollection
	if pending != nil && pending.ActorPrincipalDigest == actorPrincipalDigest &&
		pending.PlanDigest == planDigest {
		s.lifecycle.pendingCollection = nil
	}
}

func validateLifecycleCollectionTransaction(transaction lifecycleCollectionTransaction) error {
	if transaction.SchemaVersion != lifecycleCollectionSchemaVersion ||
		(transaction.State != collectionStateApplying && transaction.State != collectionStateCompleted) ||
		!digestPattern.MatchString(transaction.ActorPrincipalDigest) ||
		!digestPattern.MatchString(transaction.ReceiptDigest) ||
		transaction.Plan.SchemaVersion != lifecyclePolicySchemaVersion ||
		transaction.Plan.PolicyRevision != lifecyclePolicyRevision ||
		transaction.Plan.GeneratedAt.IsZero() ||
		!digestPattern.MatchString(transaction.Plan.PlanDigest) ||
		transaction.PlanIdentity.PolicyRevision != lifecyclePolicyRevision ||
		transaction.Next < 0 || transaction.Next > len(transaction.Plan.Candidates) ||
		len(transaction.Plan.Candidates) > maxLifecycleCollectionCandidates ||
		len(transaction.Plan.Candidates) != len(transaction.PlanIdentity.Candidates) ||
		!reflect.DeepEqual(transaction.Plan.Skipped, transaction.PlanIdentity.Skipped) ||
		!validLifecycleCollectionSkipped(transaction.Plan.Skipped) {
		return fmt.Errorf("%w: lifecycle collection transaction is invalid", ErrInvalid)
	}
	encodedIdentity, err := json.Marshal(transaction.PlanIdentity)
	if err != nil || transaction.Plan.PlanDigest != digestBytes(encodedIdentity) {
		return fmt.Errorf("%w: lifecycle collection plan identity is invalid", ErrInvalid)
	}
	var estimated int64
	previousKey := ""
	for index, candidate := range transaction.Plan.Candidates {
		identity := transaction.PlanIdentity.Candidates[index]
		key := collectionPlanItemKey(candidate)
		if !validCollectionCandidateIdentity(candidate, identity) || key <= previousKey ||
			key != collectionIdentityKey(identity) || candidate.EstimatedBytes < 0 ||
			identity.EstimatedBytes != candidate.EstimatedBytes ||
			candidate.RunID != identity.RunID || candidate.PairID != identity.PairID ||
			candidate.CampaignID != identity.CampaignID ||
			!reflect.DeepEqual(candidate.RunIDs, identity.RunIDs) ||
			candidate.RetentionClass != identity.RetentionClass ||
			!candidate.DeleteAfter.Equal(identity.DeleteAfter) ||
			!digestPattern.MatchString(identity.LifecycleDigest) ||
			!digestPattern.MatchString(identity.EvidenceDigest) {
			return fmt.Errorf("%w: lifecycle collection candidate identity is invalid", ErrInvalid)
		}
		previousKey = key
		if candidate.EstimatedBytes > 0 && estimated > int64(^uint64(0)>>1)-candidate.EstimatedBytes {
			return fmt.Errorf("%w: lifecycle collection estimate overflows", ErrInvalid)
		}
		estimated += candidate.EstimatedBytes
	}
	if estimated != transaction.Plan.EstimatedReclaimBytes ||
		!reflect.DeepEqual(transaction.Result, collectionResultThrough(transaction.Plan, transaction.Next)) ||
		(transaction.State == collectionStateCompleted) != (transaction.Next == len(transaction.Plan.Candidates)) {
		return fmt.Errorf("%w: lifecycle collection progress is invalid", ErrInvalid)
	}
	return nil
}

func validCollectionCandidateIdentity(
	candidate CollectionPlanItem,
	identity collectionItemIdentity,
) bool {
	kinds := 0
	if candidate.RunID != "" {
		kinds++
	}
	if candidate.PairID != "" {
		kinds++
	}
	if candidate.CampaignID != "" {
		kinds++
	}
	if kinds != 1 {
		return false
	}
	if candidate.DeleteAfter.IsZero() ||
		(candidate.RetentionClass != RetentionEphemeral && candidate.RetentionClass != RetentionStandard) {
		return false
	}
	switch {
	case candidate.RunID != "":
		return validClientRequestID(candidate.RunID) && len(candidate.RunIDs) == 0 &&
			terminalStatus(identity.Status) && !identity.CompletedAt.IsZero()
	case candidate.PairID != "":
		return validClientRequestID(candidate.PairID) && len(candidate.RunIDs) == 2 &&
			validClientRequestID(candidate.RunIDs[0]) && validClientRequestID(candidate.RunIDs[1]) &&
			candidate.RunIDs[0] != candidate.RunIDs[1] && identity.Status == StatusCompleted &&
			!identity.CompletedAt.IsZero()
	default:
		return validClientRequestID(candidate.CampaignID) && len(candidate.RunIDs) == 0 &&
			identity.Status == "" && !identity.CompletedAt.IsZero()
	}
}

func validLifecycleCollectionSkipped(skipped map[string]int) bool {
	allowed := map[string]bool{
		"active": true, "held": true, "protected": true,
		"referenced": true, "not_expired": true, "batch_limit": true,
	}
	if skipped == nil || len(skipped) > len(allowed) {
		return false
	}
	for reason, count := range skipped {
		if !allowed[reason] || count < 0 {
			return false
		}
	}
	return true
}

func validateLifecycleCollectionHeader(header lifecycleCollectionHeader) error {
	if header.RecordType != "plan" || header.SchemaVersion != lifecycleCollectionSchemaVersion ||
		!digestPattern.MatchString(header.HeaderDigest) ||
		header.HeaderDigest != lifecycleCollectionHeaderDigest(header) ||
		(header.State != collectionStateApplying && header.State != collectionStateCompleted) ||
		(header.State == collectionStateCompleted) != (len(header.Plan.Candidates) == 0) {
		return fmt.Errorf("%w: lifecycle collection header is invalid", ErrInvalid)
	}
	transaction := lifecycleCollectionTransaction{
		SchemaVersion: header.SchemaVersion, State: header.State,
		ActorPrincipalDigest: header.ActorPrincipalDigest,
		Plan:                 header.Plan, PlanIdentity: header.PlanIdentity,
		Result: collectionResultThrough(header.Plan, 0), ReceiptDigest: header.HeaderDigest,
	}
	if err := validateLifecycleCollectionTransaction(transaction); err != nil {
		return err
	}
	return validateLifecycleCollectionPlanBounds(header.Plan, header.PlanIdentity)
}

func validateLifecycleCollectionProgress(
	progress lifecycleCollectionProgress,
	transaction lifecycleCollectionTransaction,
	wantNext int,
) error {
	if progress.RecordType != "progress" || progress.SchemaVersion != lifecycleCollectionSchemaVersion ||
		progress.ActorPrincipalDigest != transaction.ActorPrincipalDigest ||
		progress.PlanDigest != transaction.Plan.PlanDigest || progress.Next != wantNext ||
		progress.Next > len(transaction.Plan.Candidates) ||
		progress.PreviousDigest != transaction.ReceiptDigest ||
		!digestPattern.MatchString(progress.ProgressDigest) ||
		progress.ProgressDigest != lifecycleCollectionProgressDigest(progress) ||
		(progress.State != collectionStateApplying && progress.State != collectionStateCompleted) ||
		(progress.State == collectionStateCompleted) != (progress.Next == len(transaction.Plan.Candidates)) {
		return fmt.Errorf("%w: lifecycle collection progress chain is invalid", ErrInvalid)
	}
	return nil
}

func validateLifecycleCollectionPlanBounds(
	plan CollectionPlan,
	identity collectionPlanIdentity,
) error {
	return validateLifecycleCollectionPlanBoundsWithin(plan, identity, maxStructuredArtifactBytes)
}

func validateLifecycleCollectionPlanBoundsWithin(
	plan CollectionPlan,
	identity collectionPlanIdentity,
	maxBytes int64,
) error {
	if len(plan.Candidates) > maxLifecycleCollectionCandidates ||
		len(plan.Candidates) != len(identity.Candidates) {
		return fmt.Errorf("%w: lifecycle collection candidate count exceeds its batch bound", ErrConflict)
	}
	header := lifecycleCollectionHeader{
		RecordType: "plan", SchemaVersion: lifecycleCollectionSchemaVersion,
		ActorPrincipalDigest: digestString("collection-size-bound"), State: collectionStateApplying,
		Plan: plan, PlanIdentity: identity,
	}
	header.HeaderDigest = lifecycleCollectionHeaderDigest(header)
	encoded, err := json.Marshal(header)
	projectedBytes := int64(len(encoded) + 1 +
		len(plan.Candidates)*maxLifecycleCollectionProgressBytes)
	if err != nil || maxBytes < 1 || projectedBytes > maxBytes {
		return fmt.Errorf("%w: lifecycle collection transaction exceeds its durable byte bound", ErrConflict)
	}
	return nil
}

func lifecycleCollectionHeaderDigest(header lifecycleCollectionHeader) string {
	header.HeaderDigest = ""
	encoded, err := json.Marshal(header)
	if err != nil {
		panic(err)
	}
	return digestBytes(encoded)
}

func lifecycleCollectionProgressDigest(progress lifecycleCollectionProgress) string {
	progress.ProgressDigest = ""
	encoded, err := json.Marshal(progress)
	if err != nil {
		panic(err)
	}
	return digestBytes(encoded)
}

func decodeLifecycleCollectionRecord(data []byte, destination any) error {
	if err := rejectDuplicateJSONKeys(data); err != nil {
		return fmt.Errorf("decode lifecycle collection record: %w", err)
	}
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(destination); err != nil {
		return fmt.Errorf("decode lifecycle collection record: %w", err)
	}
	return ensureJSONEOF(decoder)
}
