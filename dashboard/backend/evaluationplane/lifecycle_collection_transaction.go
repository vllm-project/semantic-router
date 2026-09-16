package evaluationplane

import "fmt"

const (
	lifecycleCollectionSchemaVersion    = "evaluation-lifecycle-collection.v1"
	lifecycleCollectionFileName         = "collection-transaction.json"
	collectionStateApplying             = "applying"
	collectionStateCompleted            = "completed"
	maxLifecycleCollectionCandidates    = 1024
	maxLifecycleCollectionProgressBytes = 1024
	lifecycleCollectionReservedBytes    = maxStructuredArtifactBytes
)

// lifecycleCollectionTransaction is the single root-wide batch receipt. It
// never replaces the durable deletion protocol for an individual resource:
// it only freezes the authorized plan and records which independently durable
// deletions have completed.
type lifecycleCollectionTransaction struct {
	SchemaVersion        string                 `json:"schema_version"`
	State                string                 `json:"state"`
	ActorPrincipalDigest string                 `json:"actor_principal_digest"`
	Plan                 CollectionPlan         `json:"plan"`
	PlanIdentity         collectionPlanIdentity `json:"plan_identity"`
	Next                 int                    `json:"next"`
	Result               CollectionResult       `json:"result"`
	ReceiptDigest        string                 `json:"receipt_digest"`
}

type lifecycleCollectionHeader struct {
	RecordType           string                 `json:"record_type"`
	SchemaVersion        string                 `json:"schema_version"`
	ActorPrincipalDigest string                 `json:"actor_principal_digest"`
	State                string                 `json:"state"`
	Plan                 CollectionPlan         `json:"plan"`
	PlanIdentity         collectionPlanIdentity `json:"plan_identity"`
	HeaderDigest         string                 `json:"header_digest"`
}

type lifecycleCollectionProgress struct {
	RecordType           string `json:"record_type"`
	SchemaVersion        string `json:"schema_version"`
	ActorPrincipalDigest string `json:"actor_principal_digest"`
	PlanDigest           string `json:"plan_digest"`
	State                string `json:"state"`
	Next                 int    `json:"next"`
	PreviousDigest       string `json:"previous_digest"`
	ProgressDigest       string `json:"progress_digest"`
}

// pendingLifecycleCollectionProjection is process-local evidence that a
// transaction write may be visible without its durability barrier. It is
// guarded by the root coordinator's lifecycle mutex.
type pendingLifecycleCollectionProjection struct {
	ActorPrincipalDigest string
	PlanDigest           string
	Transaction          lifecycleCollectionTransaction
}

type collectionExecutionHooks struct {
	active           func([]string) bool
	closeSubscribers func([]string)
}

func (s *Store) collectLifecycleUnlocked(
	actor Actor,
	request CollectionRequest,
	hooks collectionExecutionHooks,
) (CollectionResult, error) {
	if !request.Apply && request.PlanDigest != "" {
		return CollectionResult{}, fmt.Errorf("%w: dry-run collection cannot supply plan_digest", ErrInvalid)
	}
	if request.Apply && !digestPattern.MatchString(request.PlanDigest) {
		return CollectionResult{}, fmt.Errorf("%w: collection plan is stale or does not match", ErrConflict)
	}
	result, handled, err := s.resumePendingLifecycleCollectionUnlocked(actor, request, hooks)
	if err != nil || handled {
		return result, err
	}

	transaction, exists, err := s.readLifecycleCollectionTransactionUnlocked()
	if err != nil {
		return CollectionResult{}, err
	}
	if exists && transaction.State == collectionStateApplying {
		if !request.Apply {
			return CollectionResult{}, fmt.Errorf("%w: lifecycle collection is already in progress", ErrConflict)
		}
		if transaction.Plan.PlanDigest != request.PlanDigest ||
			transaction.ActorPrincipalDigest != actor.principalDigest {
			return CollectionResult{}, fmt.Errorf("%w: lifecycle collection retry does not match the active transaction", ErrConflict)
		}
		transaction, err = s.resolveLifecycleCollectionTransactionUnlocked()
		if err != nil {
			return CollectionResult{}, err
		}
		return s.resumeLifecycleCollectionUnlocked(actor, &transaction, hooks)
	}
	if exists && request.Apply && transaction.Plan.PlanDigest == request.PlanDigest {
		if transaction.ActorPrincipalDigest != actor.principalDigest {
			return CollectionResult{}, fmt.Errorf("%w: lifecycle collection receipt belongs to another actor", ErrConflict)
		}
		transaction, err = s.resolveLifecycleCollectionTransactionUnlocked()
		if err != nil {
			return CollectionResult{}, err
		}
		return transaction.Result, nil
	}
	return s.startLifecycleCollectionUnlocked(actor, request, hooks)
}

func (s *Store) resumePendingLifecycleCollectionUnlocked(
	actor Actor,
	request CollectionRequest,
	hooks collectionExecutionHooks,
) (CollectionResult, bool, error) {
	pending := s.lifecycle.pendingCollection
	if pending == nil {
		return CollectionResult{}, false, nil
	}
	if !request.Apply || pending.ActorPrincipalDigest != actor.principalDigest ||
		pending.PlanDigest != request.PlanDigest {
		return CollectionResult{}, true, fmt.Errorf(
			"%w: lifecycle collection durability retry does not match the active transaction", ErrConflict,
		)
	}
	exists, err := collectionPathExists(s.lifecycleCollectionPath())
	if err != nil {
		return CollectionResult{}, true, err
	}
	if !exists {
		s.clearPendingLifecycleCollectionUnlocked(pending.ActorPrincipalDigest, pending.PlanDigest)
		return CollectionResult{}, false, nil
	}
	transaction, err := s.resolveLifecycleCollectionTransactionUnlocked()
	if err != nil {
		return CollectionResult{}, true, err
	}
	if transaction.ActorPrincipalDigest != pending.ActorPrincipalDigest ||
		transaction.Plan.PlanDigest != pending.PlanDigest {
		if transaction.State == collectionStateCompleted &&
			transaction.Plan.PlanDigest != pending.PlanDigest {
			// The immutable old receipt survived a failed header publication.
			// Resolve established its durability, so the matching caller can
			// discard only the process-local uncertainty and publish the exact
			// plan again. Applying or same-plan mismatches remain corruption.
			result, retryErr := s.retryUnpublishedLifecycleCollectionHeaderUnlocked(
				actor,
				pending,
				hooks,
			)
			return result, true, retryErr
		}
		return CollectionResult{}, true, fmt.Errorf("%w: lifecycle collection pending projection is invalid", ErrInvalid)
	}
	s.clearPendingLifecycleCollectionUnlocked(pending.ActorPrincipalDigest, pending.PlanDigest)
	if transaction.State == collectionStateCompleted {
		return transaction.Result, true, nil
	}
	result, err := s.resumeLifecycleCollectionUnlocked(actor, &transaction, hooks)
	return result, true, err
}

func (s *Store) retryUnpublishedLifecycleCollectionHeaderUnlocked(
	actor Actor,
	pending *pendingLifecycleCollectionProjection,
	hooks collectionExecutionHooks,
) (CollectionResult, error) {
	retry := pending.Transaction
	if retry.ReceiptDigest != "" || retry.ActorPrincipalDigest != pending.ActorPrincipalDigest ||
		retry.Plan.PlanDigest != pending.PlanDigest {
		return CollectionResult{}, fmt.Errorf("%w: lifecycle collection pending projection is invalid", ErrInvalid)
	}
	s.clearPendingLifecycleCollectionUnlocked(pending.ActorPrincipalDigest, pending.PlanDigest)
	if err := s.persistLifecycleCollectionTransactionUnlocked(&retry); err != nil {
		return CollectionResult{}, err
	}
	if retry.State == collectionStateCompleted {
		return retry.Result, nil
	}
	return s.resumeLifecycleCollectionUnlocked(actor, &retry, hooks)
}

func (s *Store) startLifecycleCollectionUnlocked(
	actor Actor,
	request CollectionRequest,
	hooks collectionExecutionHooks,
) (CollectionResult, error) {
	plan, identity, err := s.buildCollectionPlanIdentityUnlocked()
	if err != nil {
		return CollectionResult{}, err
	}
	if !request.Apply {
		return CollectionResult{
			SchemaVersion: lifecyclePolicySchemaVersion,
			Applied:       false,
			Plan:          plan,
			DeletedRunIDs: []string{}, DeletedPairIDs: []string{}, DeletedCampaignIDs: []string{},
		}, nil
	}
	if request.PlanDigest != plan.PlanDigest {
		return CollectionResult{}, fmt.Errorf("%w: collection plan is stale or does not match", ErrConflict)
	}
	if err := s.requireLifecycleCollectionReservationUnlocked(); err != nil {
		return CollectionResult{}, err
	}
	transaction := newLifecycleCollectionTransaction(actor, plan, identity)
	if err := s.persistLifecycleCollectionTransactionUnlocked(&transaction); err != nil {
		return CollectionResult{}, err
	}
	if transaction.State == collectionStateCompleted {
		return transaction.Result, nil
	}
	return s.resumeLifecycleCollectionUnlocked(actor, &transaction, hooks)
}

func (s *Store) requireLifecycleCollectionReservationUnlocked() error {
	s.lifecycle.evidenceMu.Lock()
	defer s.lifecycle.evidenceMu.Unlock()
	s.runIndex.coordinator.Lock()
	defer s.runIndex.coordinator.Unlock()
	s.mu.Lock()
	defer s.mu.Unlock()
	snapshot, err := s.lifecycleUsageUnlocked()
	if err != nil {
		return err
	}
	if snapshot.report.ChargeableBytes > snapshot.report.MaxStoreBytes {
		return fmt.Errorf("%w: evaluation store has no reserved lifecycle collection capacity", ErrQuota)
	}
	return nil
}

func newLifecycleCollectionTransaction(
	actor Actor,
	plan CollectionPlan,
	identity collectionPlanIdentity,
) lifecycleCollectionTransaction {
	state := collectionStateApplying
	if len(plan.Candidates) == 0 {
		state = collectionStateCompleted
	}
	return lifecycleCollectionTransaction{
		SchemaVersion:        lifecycleCollectionSchemaVersion,
		State:                state,
		ActorPrincipalDigest: actor.principalDigest,
		Plan:                 plan,
		PlanIdentity:         identity,
		Result:               collectionResultThrough(plan, 0),
	}
}
