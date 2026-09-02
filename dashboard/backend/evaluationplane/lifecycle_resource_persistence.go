package evaluationplane

import (
	"fmt"
	"reflect"
)

type pendingLifecycleMutation struct {
	actorDigest   string
	requestDigest string
	valueDigest   string
}

func lifecycleMutationProjectionIdentity(resource lifecycleResourceRef) string {
	return "lifecycle-mutation:" + resource.Kind + ":" + resource.ID
}

// writeRunLifecycleResource and writeCampaignLifecycleResource keep the
// process-local commit projection conservative across an atomic rename whose
// containing-directory sync failed. The projection is bounded by the durable
// resource inventory and shared by every Store on the same root.
func (s *Store) writeRunLifecycleResource(
	actor Actor,
	request UpdateLifecycleRequest,
	path string,
	lifecycle RunLifecycle,
) error {
	resource := lifecycleResourceRef{Kind: lifecycleResourceRun, ID: lifecycle.RunID}
	return s.writeLifecycleResource(
		resource,
		actor.principalDigest,
		lifecycleDigest(request),
		lifecycle.PolicyDigest,
		path,
		lifecycle,
		func() (bool, error) {
			var visible RunLifecycle
			err := readJSON(path, &visible)
			return err == nil && reflect.DeepEqual(visible, lifecycle), err
		},
	)
}

func (s *Store) writeCampaignLifecycleResource(
	actor Actor,
	request UpdateLifecycleRequest,
	path string,
	lifecycle CampaignLifecycle,
) error {
	resource := lifecycleResourceRef{Kind: lifecycleResourceCampaign, ID: lifecycle.CampaignID}
	return s.writeLifecycleResource(
		resource,
		actor.principalDigest,
		lifecycleDigest(request),
		lifecycle.PolicyDigest,
		path,
		lifecycle,
		func() (bool, error) {
			var visible CampaignLifecycle
			err := readJSON(path, &visible)
			return err == nil && reflect.DeepEqual(visible, lifecycle), err
		},
	)
}

func (s *Store) writeLifecycleResource(
	resource lifecycleResourceRef,
	actorDigest string,
	requestDigest string,
	valueDigest string,
	path string,
	value any,
	visible func() (bool, error),
) error {
	s.runIndex.coordinator.Lock()
	defer s.runIndex.coordinator.Unlock()
	s.lifecycle.lifecycleResourceMu.Lock()
	defer s.lifecycle.lifecycleResourceMu.Unlock()
	s.lifecycle.pendingLifecycle[resource] = pendingLifecycleMutation{
		actorDigest:   actorDigest,
		requestDigest: requestDigest,
		valueDigest:   valueDigest,
	}
	s.runIndex.markPendingChange(lifecycleMutationProjectionIdentity(resource))
	err := s.lifecyclePersistence.Write(path, value)
	if err == nil {
		delete(s.lifecycle.pendingLifecycle, resource)
		s.runIndex.clearPendingChange(lifecycleMutationProjectionIdentity(resource))
		return nil
	}
	// A failure before rename leaves the previous canonical value intact and
	// creates no durability uncertainty. A visible desired value means rename
	// completed and must remain pending until this directory is synced.
	if published, readErr := visible(); readErr == nil && !published {
		delete(s.lifecycle.pendingLifecycle, resource)
		s.runIndex.clearPendingChange(lifecycleMutationProjectionIdentity(resource))
	}
	return err
}

func (s *Store) resolveLifecycleResourceDurability(
	resource lifecycleResourceRef,
	directory string,
	actorDigest string,
	requestDigest string,
	valueDigest string,
) error {
	s.runIndex.coordinator.Lock()
	defer s.runIndex.coordinator.Unlock()
	s.lifecycle.lifecycleResourceMu.Lock()
	defer s.lifecycle.lifecycleResourceMu.Unlock()
	pending, exists := s.lifecycle.pendingLifecycle[resource]
	if !exists {
		return nil
	}
	if pending.actorDigest != actorDigest || pending.requestDigest != requestDigest ||
		pending.valueDigest != valueDigest {
		return fmt.Errorf("%w: evaluation lifecycle mutation retry does not match the pending operation", ErrConflict)
	}
	if err := s.lifecyclePersistence.SyncDirectory(
		directory, "evaluation "+resource.Kind+" lifecycle commit retry",
	); err != nil {
		return fmt.Errorf("%w: evaluation lifecycle durability is uncertain: %w", ErrConflict, err)
	}
	delete(s.lifecycle.pendingLifecycle, resource)
	s.runIndex.clearPendingChange(lifecycleMutationProjectionIdentity(resource))
	return nil
}

// reconcileUnpublishedLifecycleRetry handles the other side of a failed
// lifecycle Write: the write returned an error and the immediate canonical
// read also failed, so the process conservatively retained a pending marker,
// but this matching retry can now prove that the desired value was never
// published. Only the same actor and exact request may withdraw the marker.
func (s *Store) reconcileUnpublishedLifecycleRetry(
	resource lifecycleResourceRef,
	actorDigest string,
	requestDigest string,
	canonicalDigest string,
) error {
	s.runIndex.coordinator.Lock()
	defer s.runIndex.coordinator.Unlock()
	s.lifecycle.lifecycleResourceMu.Lock()
	defer s.lifecycle.lifecycleResourceMu.Unlock()
	pending, exists := s.lifecycle.pendingLifecycle[resource]
	if !exists || pending.valueDigest == canonicalDigest {
		return nil
	}
	if pending.actorDigest != actorDigest || pending.requestDigest != requestDigest {
		return fmt.Errorf("%w: evaluation lifecycle mutation retry does not match the pending operation", ErrConflict)
	}
	delete(s.lifecycle.pendingLifecycle, resource)
	s.runIndex.clearPendingChange(lifecycleMutationProjectionIdentity(resource))
	return nil
}

func (s *Store) requireLifecycleResourceDurable(resource lifecycleResourceRef) error {
	s.lifecycle.lifecycleResourceMu.Lock()
	defer s.lifecycle.lifecycleResourceMu.Unlock()
	if _, pending := s.lifecycle.pendingLifecycle[resource]; pending {
		return fmt.Errorf("%w: evaluation lifecycle mutation requires an explicit matching retry", ErrConflict)
	}
	return nil
}

func (s *Store) requireNoPendingLifecycleResources() error {
	s.lifecycle.lifecycleResourceMu.Lock()
	defer s.lifecycle.lifecycleResourceMu.Unlock()
	if len(s.lifecycle.pendingLifecycle) != 0 {
		return fmt.Errorf("%w: evaluation lifecycle recovery requires the startup owner or explicit mutation retry", ErrConflict)
	}
	return nil
}

func (s *Store) forgetLifecycleResourceDurability(resource lifecycleResourceRef) {
	s.lifecycle.lifecycleResourceMu.Lock()
	defer s.lifecycle.lifecycleResourceMu.Unlock()
	delete(s.lifecycle.pendingLifecycle, resource)
	s.runIndex.clearPendingChange(lifecycleMutationProjectionIdentity(resource))
}
