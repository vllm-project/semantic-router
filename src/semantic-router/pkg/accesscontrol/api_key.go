package accesscontrol

import (
	"strings"
	"time"
)

// APIKey is the stable logical credential owner. Rotating a secret creates a
// new CredentialVersion without changing this identity or its usage history.
type APIKey struct {
	NamespaceID     NamespaceID
	ID              APIKeyID
	Name            string
	Owner           SubjectRef
	ContextTeamID   TeamID
	Status          APIKeyStatus
	ExpiresAt       *time.Time
	PolicyEpoch     uint64
	DelegationEpoch uint64
	Revision        Revision
	LastUsedAt      *time.Time
	CreatedAt       time.Time
	UpdatedAt       time.Time
	DeletedAt       *time.Time
}

func (k APIKey) SubjectRef() SubjectRef {
	return SubjectRef{NamespaceID: k.NamespaceID, ID: SubjectID(k.ID), Kind: SubjectKindAPIKey}
}

func (k APIKey) Validate() error {
	return joinValidation(
		validateAPIKeyIdentity(k),
		validateAPIKeyOwner(k),
		validateAPIKeyContext(k),
		validateAPIKeyLifecycle(k),
	)
}

func validateAPIKeyOwner(k APIKey) error {
	if err := k.Owner.Validate(); err != nil {
		return invalid("owner", err.Error())
	}
	if k.Owner.NamespaceID != k.NamespaceID {
		return invalid("owner", "must be in the key namespace")
	}
	if k.Owner.Kind != SubjectKindUser && k.Owner.Kind != SubjectKindTeam {
		return invalid("owner", "must be exactly one user or team")
	}
	return nil
}

func validateAPIKeyContext(k APIKey) error {
	if k.Owner.Kind == SubjectKindTeam && k.ContextTeamID != "" {
		return invalid("context_team_id", "must be empty for a team-owned key")
	}
	return nil
}

func validateAPIKeyLifecycle(k APIKey) error {
	var statusErr, deletedErr, expiresErr error
	if !k.Status.Valid() {
		statusErr = invalid("status", "is not a valid API-key status")
	}
	if k.Status == APIKeyStatusDeleted && k.DeletedAt == nil {
		deletedErr = invalid("deleted_at", "must be set for a deleted key")
	}
	if k.Status != APIKeyStatusDeleted && k.DeletedAt != nil {
		deletedErr = invalid("deleted_at", "must be empty unless the key is deleted")
	}
	if k.DeletedAt != nil && k.DeletedAt.Before(k.CreatedAt) {
		deletedErr = joinValidation(deletedErr, invalid("deleted_at", "must not precede created_at"))
	}
	if k.ExpiresAt != nil && !k.ExpiresAt.After(k.CreatedAt) {
		expiresErr = invalid("expires_at", "must be after created_at")
	}
	return joinValidation(statusErr, deletedErr, expiresErr)
}

func validateAPIKeyIdentity(k APIKey) error {
	return joinValidation(
		validateRequired("namespace_id", string(k.NamespaceID)),
		validateRequired("id", string(k.ID)),
		validateRequired("name", k.Name),
		validatePositiveEpoch("policy_epoch", k.PolicyEpoch),
		validatePositiveEpoch("delegation_epoch", k.DelegationEpoch),
		validateRevision(k.Revision),
		validateTimestamps(k.CreatedAt, k.UpdatedAt),
	)
}

// EffectiveContextTeamID returns the trusted accounting context. Team-owned
// keys derive it from the owner; callers can never switch it with a header.
func (k APIKey) EffectiveContextTeamID() TeamID {
	if k.Owner.Kind == SubjectKindTeam {
		return TeamID(k.Owner.ID)
	}
	return k.ContextTeamID
}

// CredentialVersion stores authentication and optional reveal material for one
// independently rotatable logical-key secret.
type CredentialVersion struct {
	ID               CredentialVersionID
	APIKeyID         APIKeyID
	KID              string
	SecretHMAC       []byte
	PepperVersion    string
	SecretCiphertext []byte
	CiphertextNonce  []byte
	KEKVersion       string
	Status           CredentialStatus
	NotBefore        time.Time
	ExpiresAt        *time.Time
	RevokedAt        *time.Time
	CreatedAt        time.Time
}

func (c CredentialVersion) Validate() error {
	return joinValidation(
		validateRequired("id", string(c.ID)),
		validateRequired("api_key_id", string(c.APIKeyID)),
		validateTokenSegment("kid", c.KID),
		validateBytes("secret_hmac", c.SecretHMAC),
		validateRequired("pepper_version", c.PepperVersion),
		validateCredentialLifecycle(c),
		validateRevealEnvelope(c),
		validateCreatedAt(c.CreatedAt),
	)
}

func validateCredentialLifecycle(c CredentialVersion) error {
	var statusErr error
	if !c.Status.Valid() {
		statusErr = invalid("status", "is not a valid credential status")
	}
	return joinValidation(
		statusErr,
		validateCredentialTimeBounds(c),
		validateCredentialRevocation(c),
		validateCredentialStatusExpiry(c),
	)
}

func validateCredentialTimeBounds(c CredentialVersion) error {
	var notBeforeErr, expiresErr error
	if c.NotBefore.IsZero() {
		notBeforeErr = invalid("not_before", "must be set")
	} else if c.NotBefore.Before(c.CreatedAt) {
		notBeforeErr = invalid("not_before", "must not precede created_at")
	}
	if c.ExpiresAt != nil && !c.ExpiresAt.After(c.NotBefore) {
		expiresErr = invalid("expires_at", "must be after not_before")
	}
	return joinValidation(notBeforeErr, expiresErr)
}

func validateCredentialRevocation(c CredentialVersion) error {
	var revokedErr error
	if c.Status == CredentialStatusRevoked && c.RevokedAt == nil {
		revokedErr = invalid("revoked_at", "must be set for a revoked credential")
	}
	if c.Status != CredentialStatusRevoked && c.RevokedAt != nil {
		revokedErr = invalid("revoked_at", "must be empty unless the credential is revoked")
	}
	if c.RevokedAt != nil && c.RevokedAt.Before(c.CreatedAt) {
		revokedErr = joinValidation(revokedErr, invalid("revoked_at", "must not precede created_at"))
	}
	return revokedErr
}

func validateCredentialStatusExpiry(c CredentialVersion) error {
	if (c.Status == CredentialStatusRetiring || c.Status == CredentialStatusExpired) && c.ExpiresAt == nil {
		return invalid("expires_at", "must be set for a retiring or expired credential")
	}
	return nil
}

func validateRevealEnvelope(c CredentialVersion) error {
	hasCiphertext := len(c.SecretCiphertext) > 0
	hasNonce := len(c.CiphertextNonce) > 0
	hasKEK := strings.TrimSpace(c.KEKVersion) != ""
	if hasCiphertext != hasNonce || hasCiphertext != hasKEK {
		return invalid("secret_ciphertext", "ciphertext, nonce, and KEK version must be present together")
	}
	return nil
}

func validateTokenSegment(field, value string) error {
	if err := validateRequired(field, value); err != nil {
		return err
	}
	if len(value) < 12 || len(value) > 96 {
		return invalid(field, "must contain between 12 and 96 characters")
	}
	for _, char := range value {
		if !validTokenSegmentCharacter(char) {
			return invalid(field, "must contain only ASCII letters, digits, hyphens, or underscores")
		}
	}
	return nil
}

func validTokenSegmentCharacter(char rune) bool {
	return char >= 'a' && char <= 'z' ||
		char >= 'A' && char <= 'Z' ||
		char >= '0' && char <= '9' ||
		char == '-' || char == '_'
}

func validatePositiveEpoch(field string, epoch uint64) error {
	if epoch == 0 {
		return invalid(field, "must be positive")
	}
	return nil
}

func validateBytes(field string, value []byte) error {
	if len(value) == 0 {
		return invalid(field, "must not be empty")
	}
	return nil
}

func validateCreatedAt(createdAt time.Time) error {
	if createdAt.IsZero() {
		return invalid("created_at", "must be set")
	}
	return nil
}
