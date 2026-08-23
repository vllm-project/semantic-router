package managementserver

import (
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementapi"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementidentity"
)

func selfViewDTO(value managementidentity.SelfView) managementapi.Me {
	clusterPermissions := append([]string(nil), value.ClusterPermissions...)
	if clusterPermissions == nil {
		clusterPermissions = []string{}
	}
	namespaces := make([]managementapi.MeNamespaceScope, len(value.Namespaces))
	for index := range value.Namespaces {
		namespaces[index] = selfNamespaceDTO(value.Namespaces[index])
	}
	return managementapi.Me{
		Principal: managementapi.MePrincipal{
			PrincipalID: string(value.Principal.Identity.ID), DisplayName: value.Principal.DisplayName,
			Kind: string(value.Session.EvidenceKind), Status: string(value.Principal.Identity.Status),
		},
		Session: managementapi.MeSession{
			SessionID: value.Session.ID, AuthenticatedAt: value.Session.AuthenticatedAt,
			ExpiresAt: value.Session.ExpiresAt, EvidenceKind: string(value.Session.EvidenceKind),
		},
		ClusterPermissions: clusterPermissions,
		Namespaces:         namespaces,
	}
}

func selfNamespaceDTO(value managementidentity.SelfNamespace) managementapi.MeNamespaceScope {
	permissions := append([]string(nil), value.Permissions...)
	if permissions == nil {
		permissions = []string{}
	}
	bindings := make([]managementapi.ManagementRoleBinding, len(value.RoleBindings))
	for index := range value.RoleBindings {
		bindings[index] = bindingDTO(value.RoleBindings[index])
	}
	teams := make([]managementapi.MeTeamMembership, len(value.Teams))
	for index := range value.Teams {
		teams[index] = managementapi.MeTeamMembership{
			TeamID: value.Teams[index].TeamID, Name: value.Teams[index].Name,
			Role: value.Teams[index].Role, Status: value.Teams[index].Status,
		}
	}
	result := managementapi.MeNamespaceScope{
		Namespace: managementapi.MeNamespace{
			NamespaceID: value.ID, Name: value.Name, Status: value.Status,
			DesiredRevision: value.DesiredRevision, AppliedRevision: value.AppliedRevision,
		},
		Permissions: permissions, RoleBindings: bindings, Teams: teams,
		SelfServicePolicy: managementapi.MeSelfServicePolicy{
			MaxKeysPerUser:             value.SelfServicePolicy.MaxKeysPerUser,
			MaxDelegatedSessions:       value.SelfServicePolicy.MaxDelegatedSessions,
			DelegatedSessionTTLSeconds: value.SelfServicePolicy.DelegatedSessionTTLSeconds,
			AllowTeamKeyDelegation:     value.SelfServicePolicy.AllowTeamKeyDelegation,
			AutomaticFirstKey:          value.SelfServicePolicy.AutomaticFirstKey,
			Revision:                   value.SelfServicePolicy.Revision,
		},
	}
	if value.User != nil {
		result.User = &managementapi.MeUser{
			UserID: value.User.ID, Email: value.User.Email,
			DisplayName: value.User.DisplayName, Status: value.User.Status,
		}
	}
	return result
}

func trustedIssuerDTO(value managementidentity.TrustedIdentityIssuer) managementapi.TrustedIdentityIssuer {
	audiences := append([]string(nil), value.Audiences...)
	if audiences == nil {
		audiences = []string{}
	}
	claims := cloneDTOStringMap(value.ClaimMapping)
	assurance := cloneDTOStringMap(value.AssuranceMapping)
	return managementapi.TrustedIdentityIssuer{
		IssuerID: value.ID, Issuer: value.Issuer, Kind: string(value.Kind),
		DiscoveryURL: value.DiscoveryURL, JWKSURL: value.JWKSURL, Audiences: audiences,
		ClaimMapping: claims, AssuranceMapping: assurance, Status: string(value.Status),
		Revision: value.Revision, CreatedAt: value.CreatedAt, UpdatedAt: value.UpdatedAt,
	}
}

func cloneDTOStringMap(source map[string]string) map[string]string {
	result := make(map[string]string, len(source))
	for key, value := range source {
		result[key] = value
	}
	return result
}
