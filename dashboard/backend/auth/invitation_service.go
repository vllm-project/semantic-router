package auth

import (
	"context"
	"crypto/sha256"
	"database/sql"
	"encoding/base64"
	"errors"
	"net/mail"
	"net/url"
	"strings"
	"time"
	"unicode/utf8"

	"github.com/google/uuid"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementapi"
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
	NamespaceID    string
	IdempotencyKey string
	ExpiresInHours int
	SendEmail      bool
	CreatedBy      string
}

type invitationAcceptance struct {
	AccessToken string
	User        *User
	Onboarding  managementapi.OnboardingResult
}

func (s *Service) ConfigureInvitations(authority InvitationAuthority, mailer InvitationMailer, publicBaseURL, issuerURL string) {
	s.invitationAuthority = authority
	s.invitationMailer = mailer
	s.invitationBaseURL = strings.TrimRight(strings.TrimSpace(publicBaseURL), "/")
	s.invitationIssuerURL = strings.TrimSpace(issuerURL)
}

func (s *Service) SetInvitationAuthority(authority InvitationAuthority) {
	s.invitationAuthority = authority
}

func (s *Service) ListInvitations(ctx context.Context, actor AuthContext, namespaceID string) ([]DashboardMemberInvitation, error) {
	if s.invitationAuthority == nil {
		return nil, ErrInvitationAuthorityUnavailable
	}
	items, err := s.invitationAuthority.ListInvitations(ctx, actor, namespaceID)
	if err != nil {
		return nil, err
	}
	presentations, err := s.store.ListInvitationPresentations(ctx, namespaceID)
	if err != nil {
		return nil, err
	}
	result := make([]DashboardMemberInvitation, 0, len(items))
	for _, item := range items {
		presentation, found := presentations[item.InvitationID]
		if !found {
			presentation = invitationPresentation{
				RouterInvitationID: item.InvitationID,
				RouterNamespaceID:  item.NamespaceID, RouterRevision: item.Revision,
				Email: item.ExpectedIdentity.Email, Name: item.DisplayName,
				ExpiresAt: item.ExpiresAt.Unix(), DeliveryStatus: "router_managed",
			}
		}
		mapped, mapErr := dashboardInvitation(item, presentation)
		if mapErr != nil {
			return nil, mapErr
		}
		result = append(result, mapped)
	}
	return result, nil
}

func (s *Service) CreateInvitation(ctx context.Context, actor AuthContext, input invitationInput) (*DashboardMemberInvitation, error) {
	if s.invitationAuthority == nil || s.invitationIssuerURL == "" {
		return nil, ErrInvitationAuthorityUnavailable
	}
	input.Email = strings.ToLower(strings.TrimSpace(input.Email))
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
	grants, err := invitationRoleGrants(role)
	if err != nil {
		return nil, err
	}
	if input.ExpiresInHours <= 0 {
		input.ExpiresInHours = 7 * 24
	}
	if input.ExpiresInHours > 30*24 {
		return nil, errors.New("invitation expiry cannot exceed 30 days")
	}
	plannedSubject := uuid.NewString()
	request := managementapi.InvitationCreateRequest{
		ExpectedIdentity: managementapi.InvitationExpectedIdentity{
			Issuer: s.invitationIssuerURL, Subject: plannedSubject, Email: input.Email,
		},
		DisplayName: strings.TrimSpace(input.Name), RoleGrants: grants,
		ExpiresAt: time.Now().UTC().Add(time.Duration(input.ExpiresInHours) * time.Hour),
	}
	if request.DisplayName == "" {
		return nil, errors.New("name is required")
	}
	if input.TeamID != "" {
		teamRole := strings.TrimSpace(input.TeamRole)
		if teamRole == "" {
			teamRole = "member"
		}
		if teamRole != "member" && teamRole != "admin" {
			return nil, errors.New("team role must be member or admin")
		}
		request.Team = &managementapi.InvitationTeamAssignment{TeamID: input.TeamID, Role: teamRole}
	}
	issued, err := s.invitationAuthority.CreateInvitation(ctx, actor, input.NamespaceID, input.IdempotencyKey, request)
	if err != nil {
		return nil, err
	}
	if validationErr := validateIssuedInvitation(issued, input.NamespaceID, plannedSubject, input.Email); validationErr != nil {
		if parsed, parseErr := uuid.Parse(issued.Data.InvitationID); parseErr == nil &&
			parsed.String() == issued.Data.InvitationID && issued.Data.NamespaceID == input.NamespaceID && issued.Data.Revision > 0 {
			_, _ = s.invitationAuthority.RevokeInvitation(ctx, actor, input.NamespaceID,
				issued.Data.InvitationID, issued.Data.Revision)
		}
		return nil, validationErr
	}
	presentation, err := s.store.CreateInvitationPresentation(ctx, invitationPresentation{
		RouterInvitationID: issued.Data.InvitationID, RouterNamespaceID: issued.Data.NamespaceID,
		RouterRevision: issued.Data.Revision, Email: input.Email, Name: request.DisplayName,
		TokenDigest: invitationDigest(issued.Token), PlannedSubjectID: plannedSubject,
		ExpiresAt: issued.Data.ExpiresAt.Unix(), CreatedBy: input.CreatedBy, DeliveryStatus: "link_ready",
	})
	if err != nil {
		_, _ = s.invitationAuthority.RevokeInvitation(ctx, actor, input.NamespaceID, issued.Data.InvitationID, issued.Data.Revision)
		return nil, err
	}
	item, err := dashboardInvitation(issued.Data, *presentation)
	if err != nil {
		return nil, err
	}
	return s.deliverInvitation(ctx, &item, issued.Token, input.SendEmail)
}

func (s *Service) ResendInvitation(ctx context.Context, actor AuthContext, namespaceID, id, idempotencyKey string, expectedRevision uint64, sendEmail bool) (*DashboardMemberInvitation, error) {
	if s.invitationAuthority == nil {
		return nil, ErrInvitationAuthorityUnavailable
	}
	presentation, err := s.store.GetInvitationPresentationByID(ctx, id)
	if err != nil || presentationAvailable(presentation) != nil || presentation.RouterRevision != expectedRevision ||
		presentation.RouterNamespaceID != namespaceID {
		return nil, ErrInvitationUnavailable
	}
	expiresAt := time.Now().UTC().Add(7 * 24 * time.Hour)
	issued, err := s.invitationAuthority.RotateInvitation(ctx, actor, presentation.RouterNamespaceID,
		id, expectedRevision, idempotencyKey, &expiresAt)
	if err != nil {
		return nil, err
	}
	if validationErr := validateIssuedInvitation(issued, presentation.RouterNamespaceID, presentation.PlannedSubjectID, presentation.Email); validationErr != nil {
		return nil, validationErr
	}
	presentation, err = s.store.RotateInvitationPresentation(ctx, id, issued.Data.Revision,
		invitationDigest(issued.Token), issued.Data.ExpiresAt.Unix())
	if err != nil {
		return nil, err
	}
	item, err := dashboardInvitation(issued.Data, *presentation)
	if err != nil {
		return nil, err
	}
	return s.deliverInvitation(ctx, &item, issued.Token, sendEmail)
}

func (s *Service) RevokeInvitation(ctx context.Context, actor AuthContext, namespaceID, id string, expectedRevision uint64) (*DashboardMemberInvitation, error) {
	if s.invitationAuthority == nil {
		return nil, ErrInvitationAuthorityUnavailable
	}
	presentation, err := s.store.GetInvitationPresentationByID(ctx, id)
	if err != nil || presentationAvailable(presentation) != nil || presentation.RouterRevision != expectedRevision ||
		presentation.RouterNamespaceID != namespaceID {
		return nil, ErrInvitationUnavailable
	}
	actualRevision, err := s.invitationAuthority.RevokeInvitation(
		ctx, actor, presentation.RouterNamespaceID, id, expectedRevision,
	)
	if err != nil {
		return nil, err
	}
	if actualRevision <= expectedRevision {
		return nil, ErrInvitationAuthorityUnavailable
	}
	presentation, err = s.store.MarkInvitationRevoked(ctx, id, actualRevision)
	if err != nil {
		return nil, err
	}
	return &DashboardMemberInvitation{
		ID: id, NamespaceID: presentation.RouterNamespaceID,
		Revision: presentation.RouterRevision, Email: presentation.Email, Name: presentation.Name,
		Status: InvitationRevoked, ExpiresAt: presentation.ExpiresAt, RevokedAt: presentation.RevokedAt,
		CreatedAt: presentation.CreatedAt, CreatedBy: presentation.CreatedBy, UpdatedAt: presentation.UpdatedAt,
		LastSentAt: presentation.LastSentAt, DeliveryStatus: presentation.DeliveryStatus,
		DeliveryError: presentation.DeliveryError,
	}, nil
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
		item.DeliveryError = "Email is unavailable; copy the invitation link instead."
		_ = s.store.UpdateInvitationDelivery(ctx, item.ID, item.DeliveryStatus, item.DeliveryError)
		return item, nil
	}
	inviteURL := s.invitationBaseURL + item.InvitationPath
	if err := s.invitationMailer.SendDashboardInvitation(ctx, item.Email, inviteURL, time.Unix(item.ExpiresAt, 0)); err != nil {
		item.DeliveryStatus = "email_failed"
		item.DeliveryError = "Email delivery failed; copy the invitation link instead."
		_ = s.store.UpdateInvitationDelivery(ctx, item.ID, item.DeliveryStatus, item.DeliveryError)
		return item, nil
	}
	item.DeliveryStatus = "email_sent"
	item.DeliveryError = ""
	_ = s.store.UpdateInvitationDelivery(ctx, item.ID, item.DeliveryStatus, "")
	return item, nil
}

func (s *Service) InvitationInfo(ctx context.Context, token string) (*DashboardMemberInvitation, error) {
	item, err := s.store.GetInvitationPresentationByDigest(ctx, invitationDigest(token))
	if err != nil || presentationAvailable(item) != nil {
		return nil, ErrInvitationUnavailable
	}
	return &DashboardMemberInvitation{
		ID: item.RouterInvitationID, NamespaceID: item.RouterNamespaceID,
		Revision: item.RouterRevision, Email: item.Email, Name: item.Name, Status: InvitationPending,
		ExpiresAt: item.ExpiresAt,
	}, nil
}

func (s *Service) AcceptInvitation(ctx context.Context, token, name, password string) (*invitationAcceptance, error) {
	if s.invitationAuthority == nil {
		return nil, ErrInvitationAuthorityUnavailable
	}
	presentation, err := s.store.GetInvitationPresentationByDigest(ctx, invitationDigest(token))
	if err != nil || presentationAvailable(presentation) != nil {
		return nil, ErrInvitationUnavailable
	}
	if utf8.RuneCountInString(password) < 9 {
		return nil, errors.New("password must contain at least 9 characters")
	}
	acceptedName := strings.TrimSpace(name)
	if acceptedName == "" {
		acceptedName = strings.TrimSpace(presentation.Name)
	}
	if acceptedName == "" {
		return nil, errors.New("name is required")
	}
	hash, err := s.HashPassword(password)
	if err != nil {
		return nil, err
	}
	sessionIssuedAt := s.now().UTC().Truncate(time.Second)
	sessionExpiresAt := sessionIssuedAt.Add(s.ttlDuration)
	accepted, err := s.invitationAuthority.AcceptInvitation(ctx, RouterInvitationAcceptance{
		NamespaceID: presentation.RouterNamespaceID, InvitationToken: token,
		PlannedSubject: presentation.PlannedSubjectID, Email: presentation.Email, DisplayName: acceptedName,
		SessionExpiresAt: sessionExpiresAt,
	})
	if err != nil {
		return nil, err
	}
	if accepted.Onboarding.InvitationID != presentation.RouterInvitationID ||
		accepted.Onboarding.UserID == "" ||
		!validOptionalOnboardingKey(accepted.Onboarding, time.Now().UTC()) {
		return nil, ErrInvitationAuthorityUnavailable
	}
	now := time.Now().Unix()
	user := User{
		ID: presentation.PlannedSubjectID, Email: strings.ToLower(presentation.Email),
		Name: acceptedName, Role: accepted.DashboardRole, Status: defaultUserStatusActive,
		CreatedAt: now, UpdatedAt: now,
	}
	permissions, err := s.store.GetEffectivePermissions(ctx, user.Role, user.ID)
	if err != nil {
		return nil, err
	}
	user = *cloneSessionUser(&user, permissions)
	accessToken, session, err := s.prepareTokenAt(&user, sessionIssuedAt, sessionExpiresAt)
	if err != nil {
		return nil, err
	}
	stored, err := s.store.CompleteRouterInvitation(ctx, invitationDigest(token), user, hash, session)
	if err != nil {
		return nil, err
	}
	return &invitationAcceptance{AccessToken: accessToken, User: stored, Onboarding: accepted.Onboarding}, nil
}

func validateIssuedInvitation(issued managementapi.InvitationIssuedSecret, namespaceID, subject, email string) error {
	if issued.Token == "" || issued.Data.InvitationID == "" || issued.Data.NamespaceID != namespaceID ||
		issued.Data.ExpectedIdentity.Subject != subject || issued.Data.ExpectedIdentity.Email != email ||
		issued.Data.Revision == 0 ||
		!issued.DeliveryExpiresAt.After(time.Now().UTC()) {
		return ErrInvitationAuthorityUnavailable
	}
	return nil
}

func validOptionalOnboardingKey(result managementapi.OnboardingResult, now time.Time) bool {
	hasID, hasSecret := result.APIKeyID != "", result.APIKey != ""
	if hasID != hasSecret {
		return false
	}
	return !hasID || result.DeliveryExpiresAt.After(now)
}

func dashboardInvitation(router managementapi.Invitation, presentation invitationPresentation) (DashboardMemberInvitation, error) {
	role, err := dashboardRoleFromGrants(router.Onboarding.RoleGrants)
	if err != nil {
		return DashboardMemberInvitation{}, err
	}
	item := DashboardMemberInvitation{
		ID: router.InvitationID, NamespaceID: router.NamespaceID,
		Revision: router.Revision, Email: router.ExpectedIdentity.Email, Name: router.DisplayName,
		Role: role, Status: router.Status, ExpiresAt: router.ExpiresAt.Unix(), CreatedAt: router.CreatedAt.Unix(),
		UpdatedAt: router.UpdatedAt.Unix(), CreatedBy: presentation.CreatedBy,
		LastSentAt: presentation.LastSentAt, DeliveryStatus: presentation.DeliveryStatus,
		DeliveryError: presentation.DeliveryError,
	}
	if router.Onboarding.Team != nil {
		item.TeamID, item.TeamRole = router.Onboarding.Team.TeamID, router.Onboarding.Team.Role
	}
	if router.AcceptedAt != nil {
		value := router.AcceptedAt.Unix()
		item.AcceptedAt = &value
	}
	if router.Status == InvitationRevoked {
		item.RevokedAt = presentation.RevokedAt
	}
	return item, nil
}

func invitationDigest(token string) string {
	sum := sha256.Sum256([]byte(strings.TrimSpace(token)))
	return base64.RawURLEncoding.EncodeToString(sum[:])
}
