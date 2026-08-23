package invitationmanagement

import (
	"strings"

	"github.com/google/uuid"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscredential"
)

type APIKeyFirstKeyPreparer struct {
	peppers accesscredential.PepperKeyring
	newID   func() string
}

func NewAPIKeyFirstKeyPreparer(peppers accesscredential.PepperKeyring, newID func() string) (*APIKeyFirstKeyPreparer, error) {
	if err := peppers.Validate(); err != nil {
		return nil, ErrUnavailable
	}
	if newID == nil {
		newID = uuid.NewString
	}
	return &APIKeyFirstKeyPreparer{peppers: peppers.Clone(), newID: newID}, nil
}

// Close erases the API-key issuer's process-owned pepper material.
func (preparer *APIKeyFirstKeyPreparer) Close() {
	if preparer == nil {
		return
	}
	preparer.peppers.Close()
}

func (preparer *APIKeyFirstKeyPreparer) PrepareFirstKey(request FirstKeyRequest) (PreparedFirstKey, error) {
	if preparer == nil || !canonicalUUID(request.NamespaceID) || !canonicalUUID(request.UserID) ||
		(request.ContextTeamID != "" && !canonicalUUID(request.ContextTeamID)) || request.Now.IsZero() {
		return PreparedFirstKey{}, ErrInvalidRequest
	}
	keyID, credentialID := preparer.newID(), preparer.newID()
	if !canonicalUUID(keyID) || !canonicalUUID(credentialID) {
		return PreparedFirstKey{}, ErrUnavailable
	}
	publicID := strings.ReplaceAll(credentialID, "-", "")
	issued, err := preparer.peppers.Issue(accesscredential.KindAPIKey, publicID)
	if err != nil {
		return PreparedFirstKey{}, ErrUnavailable
	}
	now := request.Now.UTC()
	name := strings.TrimSpace(request.Name)
	if name == "" {
		name = "My API key"
	}
	key := accesscontrol.APIKey{
		NamespaceID: accesscontrol.NamespaceID(request.NamespaceID), ID: accesscontrol.APIKeyID(keyID),
		Name: name, Owner: accesscontrol.SubjectRef{
			NamespaceID: accesscontrol.NamespaceID(request.NamespaceID),
			ID:          accesscontrol.SubjectID(request.UserID), Kind: accesscontrol.SubjectKindUser,
		},
		ContextTeamID: accesscontrol.TeamID(request.ContextTeamID), Status: accesscontrol.APIKeyStatusActive,
		PolicyEpoch: 1, DelegationEpoch: 1, Revision: 1, CreatedAt: now, UpdatedAt: now,
	}
	credential := accesscontrol.CredentialVersion{
		ID: accesscontrol.CredentialVersionID(credentialID), APIKeyID: key.ID,
		KID: issued.Digest.PublicID, SecretHMAC: append([]byte(nil), issued.Digest.HMAC...),
		PepperVersion: issued.Digest.PepperVersion, Status: accesscontrol.CredentialStatusActive,
		NotBefore: now, CreatedAt: now,
	}
	if err := key.Validate(); err != nil || credential.Validate() != nil {
		return PreparedFirstKey{}, ErrUnavailable
	}
	return PreparedFirstKey{Key: key, Credential: credential, Plaintext: []byte(issued.Plaintext)}, nil
}
