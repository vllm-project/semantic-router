package evaluationplane

import (
	"fmt"
	"path/filepath"
	"testing"
)

func newStandaloneStore(root string) (*Store, error) {
	return newStoreWithLifecycleLimits(root, LifecycleLimits{})
}

func newStoreWithLifecycleLimits(root string, limits LifecycleLimits) (*Store, error) {
	absRoot, err := filepath.Abs(root)
	if err != nil {
		return nil, err
	}
	evaluationRootCoordinators.Lock()
	active := evaluationRootCoordinators.byRoot[absRoot] != nil
	evaluationRootCoordinators.Unlock()
	if active {
		return nil, fmt.Errorf("%w: standalone test Store requires every Service on the root to be closed", ErrConflict)
	}
	return newStoreWithRootCoordinator(root, limits, newEvaluationRootCoordinator(absRoot), true)
}

func newTestPeerStore(t *testing.T, source *Store) *Store {
	t.Helper()
	return newTestPeerStoreWithLifecycleLimits(t, source, LifecycleLimits{})
}

func newTestPeerStoreWithLifecycleLimits(
	t *testing.T,
	source *Store,
	limits LifecycleLimits,
) *Store {
	t.Helper()
	peer, err := openTestPeerStore(t, source, limits)
	if err != nil {
		t.Fatalf("open peer evaluation Store: %v", err)
	}
	return peer
}

func openTestPeerStore(t *testing.T, source *Store, limits LifecycleLimits) (*Store, error) {
	t.Helper()
	ownership, err := acquireEvaluationStoreOwnership(source.root)
	if err != nil {
		return nil, err
	}
	if ownership.coordinator != source.lifecycle {
		_ = ownership.release()
		return nil, fmt.Errorf("%w: peer test Store requires a live Service on the source root", ErrConflict)
	}
	var peer *Store
	err = ownership.initialize(func(startupAuthority bool) error {
		if startupAuthority {
			return fmt.Errorf("%w: peer test Store cannot become the startup owner", ErrConflict)
		}
		var openErr error
		peer, openErr = newStoreWithRootCoordinator(
			source.root,
			limits,
			ownership.coordinator,
			false,
		)
		return openErr
	})
	if err != nil {
		_ = ownership.release()
		return nil, err
	}
	t.Cleanup(func() {
		if err := ownership.release(); err != nil {
			t.Errorf("release peer evaluation Store ownership: %v", err)
		}
	})
	return peer, nil
}
