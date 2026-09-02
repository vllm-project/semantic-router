//go:build windows

package evaluationplane

import "fmt"

type evaluationStoreOwnershipLock struct{}

// Windows does not have the Unix flock contract used to protect this durable
// filesystem store. Refuse to start instead of allowing two processes to
// publish conflicting evaluation evidence.
func openEvaluationStoreOwnershipLock(string) (*evaluationStoreOwnershipLock, error) {
	return nil, fmt.Errorf("%w: evaluation store ownership locking is unsupported on Windows", ErrConflict)
}

func (lock *evaluationStoreOwnershipLock) close() error { return nil }
