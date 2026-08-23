package auth

import (
	"context"
	"errors"
	"time"
)

var ErrFirstAdminProvisioningUnavailable = errors.New("router first-administrator provisioning is unavailable")

// FirstAdminIdentity is the durable Dashboard bootstrap identity presented to
// the Router during first-install provisioning. The subject and bounded source
// session are persisted before any cross-process call. Retries reuse active
// evidence and rotate only an expired source session while preserving the
// stable subject and idempotent Router mutations.
type FirstAdminIdentity struct {
	UserID          string
	SessionID       string
	Email           string
	DisplayName     string
	AuthenticatedAt time.Time
	ExpiresAt       time.Time
}

// FirstAdminProvisioner completes Router-owned authority for the first
// Dashboard administrator. Implementations must be idempotent and may return
// only after the one-time bootstrap credential has been finalized.
type FirstAdminProvisioner interface {
	ProvisionFirstAdmin(context.Context, FirstAdminIdentity) error
}

type pendingBootstrapAdmin struct {
	User             *User
	SessionID        string
	SessionIssuedAt  time.Time
	SessionExpiresAt time.Time
}
