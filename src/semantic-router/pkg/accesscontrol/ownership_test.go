package accesscontrol

import (
	"errors"
	"strings"
	"testing"
)

func TestValidateAPIKeyRelationships(t *testing.T) {
	user := validUser()
	team := validTeam()
	membership := validMembership()

	tests := []struct {
		name          string
		key           APIKey
		relationships APIKeyRelationships
		wantErr       bool
	}{
		{
			name:          "user owner without team context",
			key:           validUserKey(),
			relationships: APIKeyRelationships{OwnerUser: &user},
		},
		{
			name: "user owner with active team context",
			key: func() APIKey {
				key := validUserKey()
				key.ContextTeamID = team.ID
				return key
			}(),
			relationships: APIKeyRelationships{
				OwnerUser: &user, ContextTeam: &team, ContextMembership: &membership,
			},
		},
		{
			name: "disabled membership is rejected",
			key: func() APIKey {
				key := validUserKey()
				key.ContextTeamID = team.ID
				return key
			}(),
			relationships: func() APIKeyRelationships {
				disabled := membership
				disabled.Status = MembershipStatusDisabled
				return APIKeyRelationships{OwnerUser: &user, ContextTeam: &team, ContextMembership: &disabled}
			}(),
			wantErr: true,
		},
		{
			name: "cross namespace owner is rejected",
			key:  validUserKey(),
			relationships: func() APIKeyRelationships {
				other := user
				other.NamespaceID = "ns-2"
				return APIKeyRelationships{OwnerUser: &other}
			}(),
			wantErr: true,
		},
		{
			name: "team owner derives context",
			key: func() APIKey {
				key := validUserKey()
				key.Owner = team.SubjectRef()
				return key
			}(),
			relationships: APIKeyRelationships{OwnerTeam: &team},
		},
		{
			name: "team owner cannot store a separate context",
			key: func() APIKey {
				key := validUserKey()
				key.Owner = team.SubjectRef()
				key.ContextTeamID = team.ID
				return key
			}(),
			relationships: APIKeyRelationships{OwnerTeam: &team},
			wantErr:       true,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			err := ValidateAPIKeyRelationships(test.key, test.relationships)
			if test.wantErr && !errors.Is(err, ErrInvalid) {
				t.Fatalf("expected validation error, got %v", err)
			}
			if !test.wantErr && err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
		})
	}
}

func TestAPIKeyEffectiveContext(t *testing.T) {
	team := validTeam()
	key := validUserKey()
	key.Owner = team.SubjectRef()
	if got := key.EffectiveContextTeamID(); got != team.ID {
		t.Fatalf("EffectiveContextTeamID() = %q, want %q", got, team.ID)
	}
}

func TestCredentialVersionValidation(t *testing.T) {
	valid := CredentialVersion{
		ID:            "credential-1",
		APIKeyID:      "key-1",
		KID:           "public-kid-1",
		SecretHMAC:    []byte("digest"),
		PepperVersion: "pepper-1",
		Status:        CredentialStatusActive,
		NotBefore:     testTime,
		CreatedAt:     testTime,
	}

	tests := []struct {
		name    string
		mutate  func(*CredentialVersion)
		wantErr bool
	}{
		{name: "non revealable", mutate: func(*CredentialVersion) {}},
		{name: "revealable", mutate: func(c *CredentialVersion) {
			c.SecretCiphertext = []byte("ciphertext")
			c.CiphertextNonce = []byte("nonce")
			c.KEKVersion = "kek-1"
		}},
		{name: "partial envelope", mutate: func(c *CredentialVersion) {
			c.SecretCiphertext = []byte("ciphertext")
		}, wantErr: true},
		{name: "revoked without timestamp", mutate: func(c *CredentialVersion) {
			c.Status = CredentialStatusRevoked
		}, wantErr: true},
		{name: "invalid kid character", mutate: func(c *CredentialVersion) {
			c.KID = "invalid.kid.value"
		}, wantErr: true},
		{name: "maximum supported kid alphabet and length", mutate: func(c *CredentialVersion) {
			c.KID = "kid_" + strings.Repeat("a", 92)
		}},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			credential := valid
			test.mutate(&credential)
			err := credential.Validate()
			if test.wantErr && !errors.Is(err, ErrInvalid) {
				t.Fatalf("expected validation error, got %v", err)
			}
			if !test.wantErr && err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
		})
	}
}
