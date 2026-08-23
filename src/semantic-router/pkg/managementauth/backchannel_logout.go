package managementauth

import (
	"context"
	"time"
)

// BackchannelLogoutIdentity is the verified, non-secret logout selector. The
// raw signed token and plaintext JTI never cross into persistence.
type BackchannelLogoutIdentity struct {
	IssuerID        string
	TokenID         string
	Subject         string
	IssuerSessionID string
	IssuedAt        time.Time
	ExpiresAt       time.Time
	ClaimsDigest    [32]byte
}

type BackchannelLogoutVerifier interface {
	VerifyBackchannelLogout(
		context.Context,
		string,
		string,
		time.Time,
	) (BackchannelLogoutIdentity, error)
}
