package auth

import (
	"context"
	"crypto/rand"
	"crypto/sha256"
	"database/sql"
	"encoding/base64"
	"errors"
	"fmt"
	"net/mail"
	"net/url"
	"strings"
	"time"
	"unicode/utf8"

	"github.com/google/uuid"
)

var ErrInvitationUnavailable = errors.New("invitation is invalid, expired, or already used")

type InvitationMailer interface {
	Configured() bool
	SendDashboardInvitation(ctx context.Context, recipient, invitationURL string, expiresAt time.Time) error
}

type invitationInput struct {
	Email          string
	Name           string
	Role           string
	TeamID         string
	TeamRole       string
	ExpiresInHours int
	SendEmail      bool
	CreatedBy      string
}

func (s *Service) ConfigureInvitations(mailer InvitationMailer, publicBaseURL string) {
	s.invitationMailer = mailer
	s.invitationBaseURL = strings.TrimRight(strings.TrimSpace(publicBaseURL), "/")
}

func (s *Service) CreateInvitation(ctx context.Context, input invitationInput) (*DashboardMemberInvitation, error) {
	input.Email = strings.ToLower(strings.TrimSpace(input.Email))
	input.TeamID = strings.TrimSpace(input.TeamID)
	address, err := mail.ParseAddress(input.Email)
	if err != nil || address.Address != input.Email {
		return nil, errors.New("a valid email is required")
	}
	if _, _, _, _, _, _, _, _, _, lookupErr := s.store.GetUserByEmail(ctx, input.Email); lookupErr == nil {
		return nil, errors.New("a dashboard member with this email already exists")
	} else if !errors.Is(lookupErr, sql.ErrNoRows) {
		return nil, lookupErr
	}
	role, err := normalizeRole(input.Role)
	if err != nil {
		return nil, err
	}
	if role == "" {
		role = RoleRead
	}
	input.TeamRole = strings.ToLower(strings.TrimSpace(input.TeamRole))
	if input.TeamID == "" && input.TeamRole != "" {
		return nil, errors.New("team role requires a Team")
	}
	if input.TeamID != "" && input.TeamRole == "" {
		input.TeamRole = "member"
	}
	if input.TeamID != "" && input.TeamRole != "admin" && input.TeamRole != "member" {
		return nil, errors.New("team role must be Admin or Member")
	}
	if input.TeamID != "" && s.modelTeamName(ctx, input.TeamID) == "" {
		return nil, errors.New("selected Team does not exist")
	}
	if input.ExpiresInHours <= 0 {
		input.ExpiresInHours = 7 * 24
	}
	if input.ExpiresInHours > 30*24 {
		return nil, errors.New("invitation expiry cannot exceed 30 days")
	}
	token, digest, err := newInvitationToken()
	if err != nil {
		return nil, err
	}
	item, err := s.store.CreateInvitation(ctx, DashboardMemberInvitation{
		Email: input.Email, Name: strings.TrimSpace(input.Name), Role: role,
		TeamID: input.TeamID, TeamRole: input.TeamRole, CreatedBy: input.CreatedBy,
		ExpiresAt:      time.Now().Add(time.Duration(input.ExpiresInHours) * time.Hour).Unix(),
		DeliveryStatus: "link_ready",
	}, digest)
	if err != nil {
		return nil, err
	}
	return s.deliverInvitation(ctx, item, token, input.SendEmail)
}

func (s *Service) ResendInvitation(ctx context.Context, id string, sendEmail bool) (*DashboardMemberInvitation, error) {
	token, digest, err := newInvitationToken()
	if err != nil {
		return nil, err
	}
	item, err := s.store.RotateInvitation(ctx, id, digest, time.Now().Add(7*24*time.Hour).Unix())
	if err != nil {
		return nil, err
	}
	return s.deliverInvitation(ctx, item, token, sendEmail)
}

func (s *Service) deliverInvitation(ctx context.Context, item *DashboardMemberInvitation, token string, sendEmail bool) (*DashboardMemberInvitation, error) {
	item.InvitationToken = token
	item.InvitationPath = "/login?invite=1&token=" + url.QueryEscape(token)
	if !sendEmail {
		item.DeliveryStatus = "link_ready"
		_ = s.store.UpdateInvitationDelivery(ctx, item.ID, item.DeliveryStatus, "")
		return item, nil
	}
	if s.invitationMailer == nil || !s.invitationMailer.Configured() || s.invitationBaseURL == "" {
		item.DeliveryStatus = "email_not_configured"
		item.DeliveryError = "SMTP and DASHBOARD_PUBLIC_URL must be configured; copy the invitation link instead"
		_ = s.store.UpdateInvitationDelivery(ctx, item.ID, item.DeliveryStatus, item.DeliveryError)
		return item, nil
	}
	inviteURL := s.invitationBaseURL + item.InvitationPath
	if err := s.invitationMailer.SendDashboardInvitation(ctx, item.Email, inviteURL, time.Unix(item.ExpiresAt, 0)); err != nil {
		item.DeliveryStatus = "email_failed"
		item.DeliveryError = "email delivery failed; copy the invitation link instead"
		_ = s.store.UpdateInvitationDelivery(ctx, item.ID, item.DeliveryStatus, item.DeliveryError)
		return item, nil
	}
	item.DeliveryStatus = "email_sent"
	item.DeliveryError = ""
	_ = s.store.UpdateInvitationDelivery(ctx, item.ID, item.DeliveryStatus, "")
	return item, nil
}

func (s *Service) InvitationInfo(ctx context.Context, token string) (*DashboardMemberInvitation, error) {
	item, err := s.store.GetInvitationByDigest(ctx, invitationDigest(token))
	if err != nil || item.Status != InvitationPending {
		return nil, ErrInvitationUnavailable
	}
	return item, nil
}

func (s *Service) AcceptInvitation(ctx context.Context, token, name, password string) (string, *User, error) {
	invitation, err := s.InvitationInfo(ctx, token)
	if err != nil {
		return "", nil, err
	}
	if utf8.RuneCountInString(password) < 9 {
		return "", nil, errors.New("password must contain at least 9 characters")
	}
	hash, err := s.HashPassword(password)
	if err != nil {
		return "", nil, err
	}
	acceptedName := strings.TrimSpace(name)
	if acceptedName == "" {
		acceptedName = strings.TrimSpace(invitation.Name)
	}
	if acceptedName == "" {
		return "", nil, errors.New("name is required")
	}
	userID := uuid.NewString()
	teamID := strings.TrimSpace(invitation.TeamID)
	if provisionErr := s.provisionModelUser(
		ctx,
		userID,
		invitation.Email,
		acceptedName,
		&teamID,
		invitation.TeamRole,
	); provisionErr != nil {
		return "", nil, fmt.Errorf("prepare model access: %w", provisionErr)
	}
	user, err := s.store.AcceptInvitation(ctx, invitationDigest(token), userID, hash, acceptedName)
	if err != nil {
		_ = s.removeModelUser(ctx, userID)
		return "", nil, err
	}
	accessToken, err := s.issueTokenForContext(ctx, user)
	if err != nil {
		return "", nil, err
	}
	return accessToken, user, nil
}

func newInvitationToken() (string, string, error) {
	raw := make([]byte, 32)
	if _, err := rand.Read(raw); err != nil {
		return "", "", fmt.Errorf("generate invitation token: %w", err)
	}
	token := base64.RawURLEncoding.EncodeToString(raw)
	return token, invitationDigest(token), nil
}

func invitationDigest(token string) string {
	sum := sha256.Sum256([]byte(strings.TrimSpace(token)))
	return base64.RawURLEncoding.EncodeToString(sum[:])
}
