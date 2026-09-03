package evaluationplane

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"time"
)

type lifecyclePolicyPersistence interface {
	Write(path string, value any) error
	SyncDirectory(path, description string) error
}

type atomicLifecyclePolicyPersistence struct{}

func (atomicLifecyclePolicyPersistence) Write(path string, value any) error {
	return writeJSONAtomic(path, value)
}

func (atomicLifecyclePolicyPersistence) SyncDirectory(path, description string) error {
	return syncEvaluationDirectory(path, description)
}

func (s *Store) syncLifecycleResourceDirectory(directory, description string) error {
	if err := s.lifecyclePersistence.SyncDirectory(directory, description); err != nil {
		return fmt.Errorf("evaluation lifecycle durability is uncertain: %w", err)
	}
	return nil
}

func (s *Store) initializeLifecyclePolicyUnlocked(requestedLimits LifecycleLimits) error {
	explicit := requestedLimits != (LifecycleLimits{})
	var requested lifecycleStorePolicy
	if explicit {
		limits, err := normalizeLifecycleLimits(requestedLimits)
		if err != nil {
			return err
		}
		requested = newLifecycleStorePolicy(limits)
	}
	path := filepath.Join(s.lifecycleRoot, lifecyclePolicyFileName)
	var stored lifecycleStorePolicy
	if err := readJSON(path, &stored); err != nil {
		if !errors.Is(err, ErrNotFound) {
			return fmt.Errorf("read evaluation lifecycle policy: %w", err)
		}
		empty, emptyErr := s.lifecycleStoreIsFreshUnlocked()
		if emptyErr != nil {
			return emptyErr
		}
		if !empty {
			return fmt.Errorf(
				"%w: evaluation store predates the current lifecycle contract; use a fresh store",
				ErrInvalid,
			)
		}
		if !explicit {
			requested = newLifecycleStorePolicy(DefaultLifecycleLimits())
		}
		if writeErr := writeJSONAtomic(path, requested); writeErr != nil {
			return writeErr
		}
		stored = requested
	}
	if stored.SchemaVersion != lifecyclePolicySchemaVersion || stored.PolicyRevision != lifecyclePolicyRevision {
		return fmt.Errorf(
			"%w: evaluation lifecycle policy predates %s; remove the unpublished intermediate evaluation state and start with a fresh store",
			ErrInvalid, lifecyclePolicyRevision,
		)
	}
	if err := validateLifecycleStorePolicy(stored); err != nil {
		return err
	}
	if err := s.validateDurableLifecyclePolicyCompatibilityUnlocked(stored); err != nil {
		return err
	}
	if !explicit || reflect.DeepEqual(stored, requested) {
		if err := s.lifecyclePersistence.SyncDirectory(s.lifecycleRoot, "evaluation lifecycle policy retry"); err != nil {
			return fmt.Errorf("evaluation lifecycle policy durability is uncertain: %w", err)
		}
		return s.installCommittedLifecyclePolicyUnlocked(stored)
	}
	if !lifecycleLimitsExpandMonotonically(stored.Limits, requested.Limits) {
		return fmt.Errorf("%w: configured lifecycle limits do not match the durable store policy", ErrConflict)
	}
	if err := s.lifecyclePersistence.Write(path, requested); err != nil {
		return fmt.Errorf("persist expanded evaluation lifecycle policy: %w", err)
	}
	return s.installCommittedLifecyclePolicyUnlocked(requested)
}

// validatePeerLifecyclePolicyUnlocked is a read-only opener check. A peer must
// use the policy already committed by the startup owner; it cannot turn a
// visible-but-unsynced policy rename into a committed expansion.
func (s *Store) validatePeerLifecyclePolicyUnlocked(requestedLimits LifecycleLimits) error {
	path := filepath.Join(s.lifecycleRoot, lifecyclePolicyFileName)
	var stored lifecycleStorePolicy
	if err := readJSON(path, &stored); err != nil {
		return fmt.Errorf("read evaluation lifecycle policy for peer: %w", err)
	}
	if err := validateLifecycleStorePolicy(stored); err != nil {
		return err
	}
	if !s.lifecycle.policyLoaded || !reflect.DeepEqual(*s.lifecyclePolicy, stored) {
		return fmt.Errorf("%w: evaluation lifecycle policy recovery requires the startup owner", ErrConflict)
	}
	if requestedLimits == (LifecycleLimits{}) {
		return nil
	}
	normalized, err := normalizeLifecycleLimits(requestedLimits)
	if err != nil {
		return err
	}
	if !reflect.DeepEqual(normalized, stored.Limits) {
		return fmt.Errorf("%w: peer lifecycle limits do not match the committed root policy", ErrConflict)
	}
	return nil
}

// validateDurableLifecyclePolicyCompatibilityUnlocked checks an on-disk
// policy without making it visible to active Stores. A visible rename is not a
// committed expansion until the lifecycle namespace sync succeeds.
func (s *Store) validateDurableLifecyclePolicyCompatibilityUnlocked(stored lifecycleStorePolicy) error {
	if !s.lifecycle.policyLoaded {
		return nil
	}
	current := *s.lifecyclePolicy
	if reflect.DeepEqual(current, stored) {
		return nil
	}
	if !lifecycleLimitsExpandMonotonically(current.Limits, stored.Limits) {
		return fmt.Errorf("%w: durable lifecycle policy conflicts with the active store policy", ErrConflict)
	}
	return nil
}

// installCommittedLifecyclePolicyUnlocked publishes a policy to every active
// Store only after its durable namespace boundary has been closed.
func (s *Store) installCommittedLifecyclePolicyUnlocked(stored lifecycleStorePolicy) error {
	if !s.lifecycle.policyLoaded {
		*s.lifecyclePolicy = stored
		s.lifecycle.policyLoaded = true
		return nil
	}
	current := *s.lifecyclePolicy
	if reflect.DeepEqual(current, stored) {
		return nil
	}
	if !lifecycleLimitsExpandMonotonically(current.Limits, stored.Limits) {
		return fmt.Errorf("%w: durable lifecycle policy conflicts with the active store policy", ErrConflict)
	}
	*s.lifecyclePolicy = stored
	return nil
}

func lifecycleLimitsExpandMonotonically(current, requested LifecycleLimits) bool {
	return requested.MaxOwnerBytes >= current.MaxOwnerBytes &&
		requested.MaxStoreBytes >= current.MaxStoreBytes &&
		requested.MaxOwnerRuns >= current.MaxOwnerRuns &&
		requested.MaxOwnerCampaigns >= current.MaxOwnerCampaigns &&
		requested.MaxAuditBytes >= current.MaxAuditBytes
}

func (s *Store) lifecycleStoreIsFreshUnlocked() (bool, error) {
	for _, directory := range []string{s.runsRoot, s.campaignRoot, s.lifecycleAuditRoot, s.lifecycleBindingRoot} {
		entries, err := os.ReadDir(directory)
		if err != nil {
			return false, err
		}
		if len(entries) != 0 {
			return false, nil
		}
	}
	return true, nil
}

func validateLifecycleStorePolicy(policy lifecycleStorePolicy) error {
	limits, err := normalizeLifecycleLimits(policy.Limits)
	if err != nil || policy.SchemaVersion != lifecyclePolicySchemaVersion ||
		policy.PolicyRevision != lifecyclePolicyRevision || policy.ReservedRunBytes != reservedRunBytes ||
		!reflect.DeepEqual(limits, policy.Limits) || policy.PolicyDigest != lifecycleDigest(policy) {
		return fmt.Errorf("%w: durable evaluation lifecycle policy is invalid", ErrInvalid)
	}
	return nil
}

func (s *Store) readRunLifecycle(run Run) (RunLifecycle, error) {
	lifecycle, err := s.readRunLifecycleEvidence(run)
	if err != nil {
		return RunLifecycle{}, err
	}
	if err := s.requireLifecycleResourceDurable(
		lifecycleResourceRef{Kind: lifecycleResourceRun, ID: run.ID},
	); err != nil {
		return RunLifecycle{}, err
	}
	return lifecycle, nil
}

func (s *Store) readRunLifecycleEvidence(run Run) (RunLifecycle, error) {
	runDir, err := s.checkedRunDir(run.ID)
	if err != nil {
		return RunLifecycle{}, err
	}
	var lifecycle RunLifecycle
	if err := readJSON(filepath.Join(runDir, lifecycleFileName), &lifecycle); err != nil {
		return RunLifecycle{}, fmt.Errorf("validate run lifecycle: %w", err)
	}
	if err := validateRunLifecycle(run, lifecycle); err != nil {
		return RunLifecycle{}, err
	}
	return lifecycle, nil
}

func (s *Store) validateLifecycleRunBindings(startupAuthority bool) error {
	s.lifecycle.mu.Lock()
	defer s.lifecycle.mu.Unlock()
	for _, run := range s.runIndex.allRuns() {
		lifecycle, err := s.readRunLifecycle(run)
		if err != nil {
			// Corrupt lifecycle metadata is represented by the run ledger's
			// quarantine warning. The store remains readable, while every
			// scientific or destructive decision fails on the incomplete ledger.
			continue
		}
		runDir, err := s.checkedRunDir(run.ID)
		if err != nil {
			return err
		}
		if startupAuthority {
			if err := s.syncLifecycleResourceDirectory(runDir, "evaluation run lifecycle startup recovery"); err != nil {
				return err
			}
		}
		record, exists := s.lifecycle.records[lifecycle.CreationAuditDigest]
		if !exists {
			record, exists = s.lifecycle.creationBindings[lifecycle.CreationAuditDigest]
		}
		if !exists || record.Action != "create" || record.Decision != "allowed" ||
			record.ResourceKind != lifecycleResourceRun || record.ResourceID != run.ID ||
			record.OwnerDigest != lifecycle.OwnerPrincipalDigest {
			return fmt.Errorf("%w: run lifecycle is not bound to its creation audit", ErrInvalid)
		}
	}
	return nil
}

func retentionDeleteAfter(class RetentionClass, now time.Time) (*time.Time, error) {
	now = now.UTC().Truncate(time.Microsecond)
	var duration time.Duration
	switch class {
	case RetentionEphemeral:
		duration = 7 * 24 * time.Hour
	case RetentionStandard:
		duration = 30 * 24 * time.Hour
	case RetentionProtected:
		return nil, nil
	default:
		return nil, fmt.Errorf("%w: unsupported retention class", ErrInvalid)
	}
	deleteAfter := now.Add(duration)
	return &deleteAfter, nil
}

func (s *Store) lifecycleForActor(actor Actor, runID string) (RunLifecycle, error) {
	if err := validateActor(actor); err != nil {
		return RunLifecycle{}, err
	}
	run, err := s.getRunUnlocked(runID)
	if err != nil {
		return RunLifecycle{}, err
	}
	lifecycle, err := s.readRunLifecycleEvidence(run)
	if err != nil {
		return RunLifecycle{}, err
	}
	if !actor.administrator && lifecycle.OwnerPrincipalDigest != actor.principalDigest {
		return RunLifecycle{}, fmt.Errorf("%w: run belongs to another evaluation principal", ErrForbidden)
	}
	if err := s.requireRunPublicationDurable(run.ID); err != nil {
		return RunLifecycle{}, err
	}
	if err := s.requireLifecycleResourceDurable(
		lifecycleResourceRef{Kind: lifecycleResourceRun, ID: run.ID},
	); err != nil {
		return RunLifecycle{}, err
	}
	return lifecycle, nil
}

func (s *Store) lifecycleForMutationActor(actor Actor, runID string) (RunLifecycle, error) {
	if err := validateActor(actor); err != nil {
		return RunLifecycle{}, err
	}
	run, err := s.getRunUnlocked(runID)
	if err != nil {
		return RunLifecycle{}, err
	}
	lifecycle, err := s.readRunLifecycleEvidence(run)
	if err != nil {
		return RunLifecycle{}, err
	}
	if !actor.administrator && lifecycle.OwnerPrincipalDigest != actor.principalDigest {
		return RunLifecycle{}, fmt.Errorf("%w: run belongs to another evaluation principal", ErrForbidden)
	}
	if err := s.requireRunPublicationDurable(run.ID); err != nil {
		return RunLifecycle{}, err
	}
	return lifecycle, nil
}

func (s *Store) RunLifecycle(actor Actor, runID string) (RunLifecycleView, error) {
	s.lifecycle.mu.Lock()
	defer s.lifecycle.mu.Unlock()
	lifecycle, lifecycleErr := s.lifecycleForActor(actor, runID)
	if lifecycleErr != nil {
		return RunLifecycleView{}, lifecycleErr
	}
	return publicRunLifecycle(lifecycle), nil
}

func (s *Store) UpdateRunLifecycle(
	actor Actor,
	runID string,
	request UpdateLifecycleRequest,
) (RunLifecycleView, error) {
	s.lifecycle.mu.Lock()
	defer s.lifecycle.mu.Unlock()
	lifecycle, lifecycleErr := s.lifecycleForMutationActor(actor, runID)
	if lifecycleErr != nil {
		if validClientRequestID(runID) && validateActor(actor) == nil {
			owner := ""
			if run, readErr := s.getRunUnlocked(runID); readErr == nil {
				if current, currentLifecycleErr := s.readRunLifecycleEvidence(run); currentLifecycleErr == nil {
					owner = current.OwnerPrincipalDigest
				}
			}
			if auditErr := s.auditLifecycleMutationDenialUnlocked(
				actor, runID, owner, request, lifecycleDenialReason(lifecycleErr),
			); auditErr != nil {
				return RunLifecycleView{}, auditErr
			}
		}
		return RunLifecycleView{}, lifecycleErr
	}
	if request.RetentionClass == nil && request.EvidenceHold == nil {
		return RunLifecycleView{}, fmt.Errorf("%w: lifecycle update contains no mutation", ErrInvalid)
	}
	now := s.lifecycleNow().UTC().Truncate(time.Microsecond)
	updated, actions, mutationErr := s.prepareRunLifecycleMutationUnlocked(actor, runID, lifecycle, request, now)
	if mutationErr != nil {
		return RunLifecycleView{}, mutationErr
	}
	if err := s.reconcileUnpublishedLifecycleRetry(
		lifecycleResourceRef{Kind: lifecycleResourceRun, ID: runID},
		actor.principalDigest,
		lifecycleDigest(request),
		lifecycle.PolicyDigest,
	); err != nil {
		return RunLifecycleView{}, err
	}
	if len(actions) == 0 {
		runDir, err := s.checkedRunDir(runID)
		if err != nil {
			return RunLifecycleView{}, err
		}
		if err := s.resolveLifecycleResourceDurability(
			lifecycleResourceRef{Kind: lifecycleResourceRun, ID: runID},
			runDir,
			actor.principalDigest,
			lifecycleDigest(request),
			lifecycle.PolicyDigest,
		); err != nil {
			return RunLifecycleView{}, err
		}
		return publicRunLifecycle(lifecycle), nil
	}
	run, runErr := s.getRunUnlocked(runID)
	if runErr != nil {
		return RunLifecycleView{}, runErr
	}
	if err := s.validateRunLifecycleMutationStateUnlocked(actor, run, lifecycle, actions); err != nil {
		return RunLifecycleView{}, err
	}
	if err := s.requireLifecycleResourceDurable(
		lifecycleResourceRef{Kind: lifecycleResourceRun, ID: runID},
	); err != nil {
		return RunLifecycleView{}, err
	}
	reason := lifecycleOwnerAuthorizationReason(actor, lifecycle.OwnerPrincipalDigest)
	for _, action := range actions {
		if _, err := s.appendLifecycleAuditUnlocked(
			actor, lifecycleResourceRun, action, "allowed", reason, runID, lifecycle.OwnerPrincipalDigest,
		); err != nil {
			return RunLifecycleView{}, err
		}
	}
	updated.UpdatedAt, updated.PolicyDigest = now, ""
	updated.PolicyDigest = lifecycleDigest(updated)
	if err := validateRunLifecycle(run, updated); err != nil {
		return RunLifecycleView{}, err
	}
	runDir, err := s.checkedRunDir(runID)
	if err != nil {
		return RunLifecycleView{}, err
	}
	if err := s.writeRunLifecycleResource(actor, request, filepath.Join(runDir, lifecycleFileName), updated); err != nil {
		return RunLifecycleView{}, err
	}
	return publicRunLifecycle(updated), nil
}

func (s *Store) prepareRunLifecycleMutationUnlocked(
	actor Actor,
	runID string,
	lifecycle RunLifecycle,
	request UpdateLifecycleRequest,
	now time.Time,
) (RunLifecycle, []string, error) {
	updated := lifecycle
	actions := make([]string, 0, 2)
	if request.RetentionClass != nil && *request.RetentionClass != lifecycle.RetentionClass {
		deleteAfter, err := retentionDeleteAfter(*request.RetentionClass, now)
		if err != nil {
			if _, auditErr := s.appendLifecycleAuditUnlocked(
				actor, lifecycleResourceRun, "retention", "denied", "invalid_request", runID, lifecycle.OwnerPrincipalDigest,
			); auditErr != nil {
				return RunLifecycle{}, nil, auditErr
			}
			return RunLifecycle{}, nil, err
		}
		updated.RetentionClass, updated.DeleteAfter = *request.RetentionClass, deleteAfter
		actions = append(actions, "retention")
	}
	if request.EvidenceHold != nil && *request.EvidenceHold != lifecycle.EvidenceHold {
		updated.EvidenceHold = *request.EvidenceHold
		if updated.EvidenceHold {
			actions = append(actions, "hold")
		} else {
			actions = append(actions, "release")
		}
	}
	return updated, actions, nil
}

func (s *Store) validateRunLifecycleMutationStateUnlocked(
	actor Actor,
	run Run,
	lifecycle RunLifecycle,
	actions []string,
) error {
	if run.Status == StatusPending || terminalStatus(run.Status) {
		return nil
	}
	if _, err := s.appendLifecycleAuditUnlocked(
		actor, lifecycleResourceRun, actions[0], "denied", "conflict", run.ID, lifecycle.OwnerPrincipalDigest,
	); err != nil {
		return err
	}
	return fmt.Errorf("%w: lifecycle policy cannot change while evaluation execution is active", ErrConflict)
}

func lifecycleOwnerAuthorizationReason(actor Actor, ownerDigest string) string {
	if actor.principalDigest == SystemActor().principalDigest {
		return "system"
	}
	if actor.administrator {
		return "administrator"
	}
	if actor.principalDigest == ownerDigest {
		return "owner"
	}
	return "not_owner"
}

func lifecycleDenialReason(err error) string {
	if isForbidden(err) {
		return "not_owner"
	}
	if errors.Is(err, ErrNotFound) {
		return "not_found"
	}
	return "conflict"
}

func isForbidden(err error) bool {
	return errors.Is(err, ErrForbidden)
}

func lifecycleMutationActions(request UpdateLifecycleRequest) []string {
	actions := make([]string, 0, 2)
	if request.RetentionClass != nil {
		actions = append(actions, "retention")
	}
	if request.EvidenceHold != nil {
		if *request.EvidenceHold {
			actions = append(actions, "hold")
		} else {
			actions = append(actions, "release")
		}
	}
	return actions
}

func (s *Store) auditLifecycleMutationDenialUnlocked(
	actor Actor,
	runID string,
	ownerDigest string,
	request UpdateLifecycleRequest,
	reason string,
) error {
	for _, action := range lifecycleMutationActions(request) {
		if _, err := s.appendLifecycleAuditUnlocked(
			actor, lifecycleResourceRun, action, "denied", reason, runID, ownerDigest,
		); err != nil {
			return err
		}
	}
	return nil
}
