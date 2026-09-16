package auth

import (
	"context"
	"crypto/rand"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"errors"
	"net/mail"
	"strings"
	"time"
)

const invitationLifetime = 72 * time.Hour

var (
	ErrInvalidInvitationEmail = errors.New("enter a valid email address")
	ErrInvalidInvitationName  = errors.New("name is required")
	ErrInvitationUserExists   = errors.New("a dashboard user with this email already exists")
	ErrInvalidInvitationKind  = errors.New("invitation kind must be personal or shared")
	ErrInvalidInvitationUses  = errors.New("shared invitation capacity must be between 2 and 100")
)

type InvitationSpec struct {
	Kind    string
	Email   string
	Name    string
	Role    string
	MaxUses int
}

func newInvitationToken() (string, string, error) {
	raw := make([]byte, 32)
	if _, err := rand.Read(raw); err != nil {
		return "", "", err
	}
	token := base64.RawURLEncoding.EncodeToString(raw)
	return token, invitationTokenDigest(token), nil
}

func invitationTokenDigest(token string) string {
	sum := sha256.Sum256([]byte(strings.TrimSpace(token)))
	return hex.EncodeToString(sum[:])
}

func validateInvitationIdentity(email, name string) (string, string, error) {
	email = strings.ToLower(strings.TrimSpace(email))
	name = strings.TrimSpace(name)
	address, err := mail.ParseAddress(email)
	if err != nil || strings.ToLower(address.Address) != email {
		return "", "", ErrInvalidInvitationEmail
	}
	if name == "" {
		return "", "", ErrInvalidInvitationName
	}
	return email, name, nil
}

func (s *Service) CreateInvitation(ctx context.Context, spec InvitationSpec, createdBy string) (*Invitation, string, error) {
	spec.Kind = strings.ToLower(strings.TrimSpace(spec.Kind))
	if spec.Kind == "" {
		spec.Kind = InvitationPersonal
	}
	if spec.Kind != InvitationPersonal && spec.Kind != InvitationShared {
		return nil, "", ErrInvalidInvitationKind
	}

	var err error
	spec.Role, err = normalizeRole(spec.Role)
	if err != nil {
		return nil, "", err
	}
	if spec.Role == "" {
		spec.Role = RoleRead
	}

	if spec.Kind == InvitationPersonal {
		spec.Email, spec.Name, err = validateInvitationIdentity(spec.Email, spec.Name)
		if err != nil {
			return nil, "", err
		}
		spec.MaxUses = 1
		exists, lookupErr := s.store.HasUserEmail(ctx, spec.Email)
		if lookupErr != nil {
			return nil, "", lookupErr
		}
		if exists {
			return nil, "", ErrInvitationUserExists
		}
	} else {
		spec.Email = ""
		spec.Name = ""
		if spec.MaxUses < 2 || spec.MaxUses > 100 {
			return nil, "", ErrInvalidInvitationUses
		}
	}

	token, digest, err := newInvitationToken()
	if err != nil {
		return nil, "", err
	}
	item, err := s.store.CreateInvitation(ctx, spec.Kind, spec.Email, spec.Name, spec.Role, digest, createdBy, spec.MaxUses, time.Now().Add(invitationLifetime).Unix())
	return item, token, err
}

func (s *Service) RotateInvitation(ctx context.Context, id string) (*Invitation, string, error) {
	token, digest, err := newInvitationToken()
	if err != nil {
		return nil, "", err
	}
	item, err := s.store.RotateInvitation(ctx, id, digest, time.Now().Add(invitationLifetime).Unix())
	return item, token, err
}

func (s *Service) InvitationInfo(ctx context.Context, token string) (*Invitation, error) {
	item, _, err := s.store.GetInvitationByDigest(ctx, invitationTokenDigest(token))
	if err != nil || item.Status != InvitationPending || item.ExpiresAt <= nowUnix() {
		return nil, ErrInvitationUnavailable
	}
	return item, nil
}

func (s *Service) AcceptInvitation(ctx context.Context, token, email, name, password string) (string, *User, error) {
	item, err := s.InvitationInfo(ctx, token)
	if err != nil {
		return "", nil, err
	}
	if item.Kind == InvitationPersonal {
		email, name = item.Email, item.Name
	} else {
		email, name, err = validateInvitationIdentity(email, name)
		if err != nil {
			return "", nil, err
		}
		exists, lookupErr := s.store.HasUserEmail(ctx, email)
		if lookupErr != nil {
			return "", nil, lookupErr
		}
		if exists {
			return "", nil, ErrInvitationUserExists
		}
	}
	hash, err := s.HashPassword(password)
	if err != nil {
		return "", nil, err
	}
	user, err := s.store.AcceptInvitation(ctx, invitationTokenDigest(token), email, name, hash)
	if err != nil {
		return "", nil, err
	}
	signed, err := s.issueTokenForContext(ctx, user)
	return signed, user, err
}
