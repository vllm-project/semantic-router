package evaluationplane

// recoverLifecycleEvidenceAndIndex is the single startup recovery barrier.
// It shares the mutation lock order used by create, evidence publication, and
// aggregate lifecycle changes, so a second Store on the same root cannot scan
// or repair a transaction while an active Store is publishing it.
func (s *Store) recoverLifecycleEvidenceAndIndex(startupAuthority bool) error {
	s.lifecycle.mu.Lock()
	defer s.lifecycle.mu.Unlock()
	s.lifecycle.evidenceMu.Lock()
	defer s.lifecycle.evidenceMu.Unlock()
	s.runIndex.coordinator.Lock()
	defer s.runIndex.coordinator.Unlock()
	s.mu.Lock()
	defer s.mu.Unlock()

	if !startupAuthority {
		if err := s.requireNoPendingRunPublications(); err != nil {
			return err
		}
		if err := s.requireNoRunDeletionIntentsUnlocked(); err != nil {
			return err
		}
		if err := s.requireNoControlledPairRecoveryTransactionsUnlocked(); err != nil {
			return err
		}
		if err := s.validateStableControlledPairsUnlocked(); err != nil {
			return err
		}
		if err := requireNoStagedRunBundles(s.runsRoot); err != nil {
			return err
		}
		return s.refreshRunIndexUnlocked()
	}
	if err := s.recoverRunDeletionsUnlocked(); err != nil {
		return err
	}
	if err := s.recoverControlledPairTransactionsUnlocked(); err != nil {
		return err
	}
	if err := recoverStagedRunBundles(s.runsRoot); err != nil {
		return err
	}
	// Close any ordinary Run publication/deletion parent uncertainty before
	// rebuilding the public index from visible directory entries.
	if err := s.recoverRunPublicationDurabilityUnlocked(); err != nil {
		return err
	}
	if err := s.recoverExecutionAttestationsUnlocked(); err != nil {
		return err
	}
	return s.refreshRunIndexUnlocked()
}
