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
)

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

func (s *Service) CreateInvitation(ctx context.Context, email, name, role, createdBy string) (*Invitation, string, error) {
	email, name, err := validateInvitationIdentity(email, name)
	if err != nil {
		return nil, "", err
	}
	role, err = normalizeRole(role)
	if err != nil {
		return nil, "", err
	}
	if role == "" {
		role = RoleRead
	}
	exists, err := s.store.HasUserEmail(ctx, email)
	if err != nil {
		return nil, "", err
	}
	if exists {
		return nil, "", ErrInvitationUserExists
	}
	token, digest, err := newInvitationToken()
	if err != nil {
		return nil, "", err
	}
	item, err := s.store.CreateInvitation(ctx, email, name, role, digest, createdBy, time.Now().Add(invitationLifetime).Unix())
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

func (s *Service) AcceptInvitation(ctx context.Context, token, password string) (string, *User, error) {
	if _, err := s.InvitationInfo(ctx, token); err != nil {
		return "", nil, err
	}
	hash, err := s.HashPassword(password)
	if err != nil {
		return "", nil, err
	}
	user, err := s.store.AcceptInvitation(ctx, invitationTokenDigest(token), hash)
	if err != nil {
		return "", nil, err
	}
	signed, err := s.issueTokenForContext(ctx, user)
	return signed, user, err
}
