package providercatalog

import (
	"errors"
	"fmt"
	"time"
)

var (
	ErrPublicationConflict = errors.New("provider catalog publication conflict")
	ErrActivationBlocked   = errors.New("provider catalog activation is blocked")
)

// PublicationState is the durable compare-and-swap token for the immutable
// Provider Catalog. Product definitions are addressed by content revision;
// generation orders desired-state mutations without exposing storage details.
type PublicationState struct {
	DesiredRevision string
	ActiveRevision  string
	Generation      uint64
	UpdatedAt       time.Time
}

type ReplicaBlocker struct {
	RolloutGroup RolloutGroup
	ReplicaID    string
	Reason       string
}

type ActivationBlockers struct {
	Missing      []RolloutGroup
	Expired      []RolloutGroup
	Divergent    []RolloutGroup
	Incompatible []ReplicaBlocker
}

func (blockers ActivationBlockers) Empty() bool {
	return len(blockers.Missing) == 0 && len(blockers.Expired) == 0 &&
		len(blockers.Divergent) == 0 && len(blockers.Incompatible) == 0
}

type ActivationBlockedError struct {
	Revision string
	Blockers ActivationBlockers
}

func (err *ActivationBlockedError) Error() string {
	if err == nil {
		return ErrActivationBlocked.Error()
	}
	return fmt.Sprintf("%s %s: missing=%d expired=%d divergent=%d incompatible=%d",
		ErrActivationBlocked, err.Revision, len(err.Blockers.Missing), len(err.Blockers.Expired),
		len(err.Blockers.Divergent), len(err.Blockers.Incompatible))
}

func (err *ActivationBlockedError) Unwrap() error { return ErrActivationBlocked }
