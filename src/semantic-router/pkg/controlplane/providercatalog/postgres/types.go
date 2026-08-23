// Package postgres persists and coordinates immutable provider catalog
// publications. Router startup reads only the PostgreSQL active revision;
// provider integrations enter through an explicit Stage operation.
package postgres

import (
	"errors"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/providercatalog"
)

var (
	ErrNoActiveSnapshot  = errors.New("provider catalog has no active snapshot")
	ErrNoDesiredSnapshot = errors.New("provider catalog has no desired snapshot")
	ErrCorruptSnapshot   = errors.New("provider catalog persisted snapshot is corrupt")
	ErrCorruptState      = errors.New("provider catalog durable state is corrupt")
	ErrStaleRevision     = errors.New("provider catalog revision is neither desired nor active")
)

type State = providercatalog.PublicationState

type StageRequest struct {
	Snapshot                *providercatalog.Snapshot
	ExpectedDesiredRevision string
	ExpectedGeneration      uint64
	RequiredRolloutGroups   []providercatalog.RolloutGroup
}

type AckStatus string

const (
	AckCompatible   AckStatus = "compatible"
	AckIncompatible AckStatus = "incompatible"
)

type AcknowledgeRequest struct {
	Revision         string
	ReplicaID        string
	RolloutGroup     providercatalog.RolloutGroup
	CapabilityDigest []byte
	Status           AckStatus
	Reason           string
	Lease            time.Duration
}

type ActivateRequest struct {
	Revision           string
	ExpectedGeneration uint64
}

type ReplicaAcknowledgement struct {
	Revision         string
	ReplicaID        string
	RolloutGroup     providercatalog.RolloutGroup
	CapabilityDigest []byte
	Status           AckStatus
	Reason           string
	AcknowledgedAt   time.Time
	LeaseExpiresAt   time.Time
}

func validRevision(value string) bool {
	if !strings.HasPrefix(value, "sha256:") || len(value) != len("sha256:")+64 {
		return false
	}
	for _, char := range strings.TrimPrefix(value, "sha256:") {
		if (char < '0' || char > '9') && (char < 'a' || char > 'f') {
			return false
		}
	}
	return true
}
