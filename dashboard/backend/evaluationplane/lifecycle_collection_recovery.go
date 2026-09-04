package evaluationplane

import "fmt"

func (s *Store) recoverLifecycleCollection(startupAuthority bool) error {
	s.lifecycle.mu.Lock()
	defer s.lifecycle.mu.Unlock()
	if startupAuthority {
		if err := recoverLifecycleCollectionTemps(s.collectionRoot); err != nil {
			return err
		}
	} else if err := requireNoLifecycleCollectionTemps(s.collectionRoot); err != nil {
		return err
	}
	if pending := s.lifecycle.pendingCollection; pending != nil {
		if !startupAuthority {
			return fmt.Errorf("%w: lifecycle collection durability recovery is required", ErrConflict)
		}
		exists, err := collectionPathExists(s.lifecycleCollectionPath())
		if err != nil {
			return err
		}
		if !exists {
			s.clearPendingLifecycleCollectionUnlocked(pending.ActorPrincipalDigest, pending.PlanDigest)
			return nil
		}
		transaction, err := s.resolveLifecycleCollectionTransactionUnlocked()
		if err != nil {
			return err
		}
		if transaction.ActorPrincipalDigest != pending.ActorPrincipalDigest ||
			transaction.Plan.PlanDigest != pending.PlanDigest {
			return fmt.Errorf("%w: lifecycle collection pending projection is invalid", ErrInvalid)
		}
		s.clearPendingLifecycleCollectionUnlocked(pending.ActorPrincipalDigest, pending.PlanDigest)
		return s.finishLifecycleCollectionRecoveryUnlocked(transaction)
	}
	transaction, exists, err := s.readLifecycleCollectionTransactionUnlocked()
	if err != nil || !exists {
		return err
	}
	if !startupAuthority {
		if transaction.State == collectionStateCompleted {
			return nil
		}
		return fmt.Errorf("%w: lifecycle collection recovery is required", ErrConflict)
	}
	transaction, err = s.resolveLifecycleCollectionTransactionUnlocked()
	if err != nil {
		return err
	}
	return s.finishLifecycleCollectionRecoveryUnlocked(transaction)
}

func (s *Store) finishLifecycleCollectionRecoveryUnlocked(
	transaction lifecycleCollectionTransaction,
) error {
	if transaction.State == collectionStateCompleted {
		return nil
	}
	actor := Actor{principalDigest: transaction.ActorPrincipalDigest, administrator: true}
	_, err := s.resumeLifecycleCollectionUnlocked(actor, &transaction, collectionExecutionHooks{})
	return err
}

func (s *Store) resumeLifecycleCollectionUnlocked(
	actor Actor,
	transaction *lifecycleCollectionTransaction,
	hooks collectionExecutionHooks,
) (CollectionResult, error) {
	if transaction.ActorPrincipalDigest != actor.principalDigest || !actor.administrator {
		return CollectionResult{}, fmt.Errorf("%w: lifecycle collection recovery actor does not match", ErrConflict)
	}
	for transaction.Next < len(transaction.Plan.Candidates) {
		index := transaction.Next
		candidate := transaction.Plan.Candidates[index]
		expected := transaction.PlanIdentity.Candidates[index]
		if err := s.executeLifecycleCollectionItemUnlocked(actor, candidate, expected, hooks); err != nil {
			return CollectionResult{}, err
		}
		transaction.Next++
		transaction.Result = collectionResultThrough(transaction.Plan, transaction.Next)
		if transaction.Next == len(transaction.Plan.Candidates) {
			transaction.State = collectionStateCompleted
		}
		if err := s.persistLifecycleCollectionTransactionUnlocked(transaction); err != nil {
			return CollectionResult{}, err
		}
	}
	return transaction.Result, nil
}
