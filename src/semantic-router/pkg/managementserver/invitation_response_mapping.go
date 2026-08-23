package managementserver

import (
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/invitationmanagement"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementapi"
)

func invitationDTO(value invitationmanagement.Invitation) managementapi.Invitation {
	grants := make([]managementapi.InvitationRoleGrant, len(value.Snapshot.RoleGrants))
	for index, grant := range value.Snapshot.RoleGrants {
		grants[index] = managementapi.InvitationRoleGrant{
			RoleID: grant.RoleID, RoleRevision: grant.RoleRevision,
			RolePermissionsDigest: grant.RolePermissionsDigest, ScopeKind: grant.ScopeKind,
			DelegationCeiling: append([]string(nil), grant.DelegationCeiling...),
			SourceBindingID:   grant.SourceBindingID, SourceBindingRevision: grant.SourceBindingRevision,
			SourceRoleID: grant.SourceRoleID, SourcePermissionsDigest: grant.SourcePermissionsDigest,
		}
	}
	var team *managementapi.InvitationTeamAssignment
	if value.Snapshot.Team != nil {
		team = &managementapi.InvitationTeamAssignment{TeamID: value.Snapshot.Team.TeamID, Role: string(value.Snapshot.Team.Role)}
	}
	return managementapi.Invitation{
		InvitationID: value.ID, NamespaceID: value.NamespaceID,
		CreatedByPrincipalID: value.CreatedByPrincipalID,
		ExpectedIdentity: managementapi.InvitationExpectedIdentity{
			Issuer:  value.Expected.Issuer,
			Subject: value.Expected.Subject, Email: value.Expected.Email,
		},
		DisplayName: value.DisplayName,
		Onboarding: managementapi.InvitationOnboardingSnapshot{
			RoleGrants: grants, Team: team,
			SelfServicePolicyRevision: value.Snapshot.SelfServicePolicyRevision,
			AccessPolicyID:            value.Snapshot.AccessPolicyID,
			AccessPolicyRevision:      value.Snapshot.AccessPolicyRevision,
			RateLimitPolicyID:         value.Snapshot.RateLimitPolicyID,
			RateLimitPolicyRevision:   value.Snapshot.RateLimitPolicyRevision,
			AutomaticFirstKey:         value.Snapshot.AutomaticFirstKey,
		},
		ExpiresAt: value.ExpiresAt, Status: string(value.Status),
		AcceptedPrincipalID: value.AcceptedPrincipalID, AcceptedUserID: value.AcceptedUserID,
		AcceptedManagementSessionID: value.AcceptedManagementSessionID,
		AcceptedAt:                  cloneResponseTime(value.AcceptedAt), Revision: value.Revision,
		CreatedAt: value.CreatedAt, UpdatedAt: value.UpdatedAt,
	}
}

func invitationPageDTO(value invitationmanagement.Page) managementapi.InvitationPage {
	items := make([]managementapi.Invitation, len(value.Items))
	for index := range value.Items {
		items[index] = invitationDTO(value.Items[index])
	}
	return managementapi.InvitationPage{Data: items, Page: managementapi.PageInfo{
		NextCursor: value.NextCursor, HasMore: value.HasMore, PageSize: value.PageSize,
	}}
}
