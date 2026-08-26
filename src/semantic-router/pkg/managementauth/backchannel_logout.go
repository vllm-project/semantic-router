package managementauth

import (
	"context"
	"crypto/sha256"
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

// IssuerSessionLogoutDigest is the durable, non-secret key for one issuer SID
// logout fence. Keeping the selector out of the tombstone table avoids
// retaining an upstream session identifier after its Management sessions age
// out.
func IssuerSessionLogoutDigest(issuerID, issuerSessionID string) [sha256.Size]byte {
	return issuerLogoutSelectorDigest("sid", issuerID, issuerSessionID)
}

// IssuerSubjectLogoutDigest is the durable, non-secret key for one issuer
// subject logout fence. The fence retains the logout iat so a genuinely later
// authentication for the same subject remains eligible for exchange.
func IssuerSubjectLogoutDigest(issuerID, subject string) [sha256.Size]byte {
	return issuerLogoutSelectorDigest("subject", issuerID, subject)
}

func issuerLogoutSelectorDigest(kind, issuerID, selector string) [sha256.Size]byte {
	return sha256.Sum256([]byte("vllm-sr/issuer-logout-selector/v1\x00" +
		kind + "\x00" + issuerID + "\x00" + selector))
}
