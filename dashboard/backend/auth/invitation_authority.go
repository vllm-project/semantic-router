package auth

import (
	"context"
	"errors"
	"net/http"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementapi"
)

const (
	routerPlatformAdminRoleID = "10000000-0000-5000-8000-000000000002"
	routerOperatorRoleID      = "10000000-0000-5000-8000-000000000003"
	routerViewerRoleID        = "10000000-0000-5000-8000-000000000007"
	routerConsumerRoleID      = "10000000-0000-5000-8000-000000000008"
)

var ErrInvitationAuthorityUnavailable = errors.New("router invitation authority is unavailable")

// InvitationAuthority is the sole identity and access authority behind the
// Dashboard invitation BFF. The Dashboard store may retain presentation
// metadata, but never implements this interface itself.
type InvitationAuthority interface {
	ListInvitations(context.Context, AuthContext, string) ([]managementapi.Invitation, error)
	CreateInvitation(context.Context, AuthContext, string, string, managementapi.InvitationCreateRequest) (managementapi.InvitationIssuedSecret, error)
	RotateInvitation(context.Context, AuthContext, string, string, uint64, string, *time.Time) (managementapi.InvitationIssuedSecret, error)
	RevokeInvitation(context.Context, AuthContext, string, string, uint64) (uint64, error)
	AcceptInvitation(context.Context, RouterInvitationAcceptance) (RouterInvitationAcceptanceResult, error)
}

type RouterInvitationAcceptance struct {
	NamespaceID      string
	InvitationToken  string
	PlannedSubject   string
	Email            string
	DisplayName      string
	SessionExpiresAt time.Time
}

type RouterInvitationAcceptanceResult struct {
	Onboarding    managementapi.OnboardingResult
	DashboardRole string
}

// InvitationAuthorityError preserves only the public Router HTTP status. It
// deliberately omits upstream response bodies and secret-bearing requests.
type InvitationAuthorityError struct {
	Status int
}

func (err *InvitationAuthorityError) Error() string {
	if err == nil {
		return ErrInvitationAuthorityUnavailable.Error()
	}
	return http.StatusText(err.Status)
}

func invitationRoleGrants(role string) ([]managementapi.InvitationRoleGrantRequest, error) {
	roleID := map[string]string{
		RoleAdmin: routerPlatformAdminRoleID,
		RoleWrite: routerOperatorRoleID,
		RoleRead:  routerViewerRoleID,
	}[role]
	if roleID == "" {
		return nil, errors.New("invalid Dashboard role")
	}
	return []managementapi.InvitationRoleGrantRequest{
		{RoleID: roleID, ScopeKind: "namespace"},
		{RoleID: routerConsumerRoleID, ScopeKind: "user"},
	}, nil
}

func dashboardRoleFromGrants(grants []managementapi.InvitationRoleGrant) (string, error) {
	roleIDs := make([]string, 0, len(grants))
	for _, grant := range grants {
		if grant.ScopeKind == "namespace" {
			roleIDs = append(roleIDs, grant.RoleID)
		}
	}
	return DashboardRoleFromManagementRoleIDs(roleIDs)
}

func DashboardRoleFromManagementRoleIDs(roleIDs []string) (string, error) {
	rank := map[string]int{routerViewerRoleID: 1, routerOperatorRoleID: 2, routerPlatformAdminRoleID: 3}
	roles := map[int]string{1: RoleRead, 2: RoleWrite, 3: RoleAdmin}
	selected := 0
	for _, roleID := range roleIDs {
		if rank[roleID] > selected {
			selected = rank[roleID]
		}
	}
	if selected == 0 {
		return "", ErrInvitationAuthorityUnavailable
	}
	return roles[selected], nil
}
