package managementserver

import (
	"slices"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accessmanagement"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementapi"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauth"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/namespacemanagement"
)

func namespaceDTO(value namespacemanagement.Namespace) managementapi.Namespace {
	return managementapi.Namespace{
		NamespaceID: value.ID, Name: value.Name, QuotaPartitionID: value.QuotaPartitionID,
		BillingCurrency: value.BillingCurrency, Status: string(value.Status), Revision: value.Revision,
		RuntimeEpoch: value.RuntimeEpoch, CreatedAt: value.CreatedAt, UpdatedAt: value.UpdatedAt,
	}
}

func namespacePageDTO(value namespacemanagement.Page[namespacemanagement.Namespace]) managementapi.NamespacePage {
	items := make([]managementapi.Namespace, len(value.Items))
	for index := range value.Items {
		items[index] = namespaceDTO(value.Items[index])
	}
	return managementapi.NamespacePage{Data: items, Page: managementapi.PageInfo{NextCursor: value.NextCursor, HasMore: value.HasMore, PageSize: value.PageSize}}
}

func selfServicePolicyDTO(value namespacemanagement.SelfServicePolicy) managementapi.SelfServicePolicy {
	capabilities := make([]string, len(value.TeamAdminCapabilities))
	for index := range value.TeamAdminCapabilities {
		capabilities[index] = string(value.TeamAdminCapabilities[index])
	}
	return managementapi.SelfServicePolicy{
		NamespaceID: value.NamespaceID, MaxKeysPerUser: value.MaxKeysPerUser,
		MaxDelegatedSessions: value.MaxDelegatedSessions, DelegatedSessionTTLSeconds: int64(value.DelegatedSessionTTL / time.Second),
		AllowTeamKeyDelegation: value.AllowTeamKeyDelegation, AutomaticFirstKey: value.AutomaticFirstKey,
		TeamAdminCapabilities: capabilities, DefaultAccessPolicyID: value.DefaultAccessPolicyID,
		DefaultRateLimitPolicyID: value.DefaultRateLimitPolicyID, Revision: value.Revision, SeedVersion: value.SeedVersion, UpdatedAt: value.UpdatedAt,
	}
}

func securityPolicyDTO(value namespacemanagement.ManagementSecurityPolicy) managementapi.NamespaceManagementSecurityPolicy {
	return managementapi.NamespaceManagementSecurityPolicy{
		NamespaceID:        value.NamespaceID,
		ActionRequirements: requirementsDTO(value.ActionRequirements), SeedVersion: value.SeedVersion,
		Revision: value.Revision, UpdatedAt: value.UpdatedAt,
	}
}

func requirementsDTO(source map[string]managementauth.ActionRequirement) map[string][]managementapi.AuthenticationRequirement {
	result := make(map[string][]managementapi.AuthenticationRequirement, len(source))
	for action, requirement := range source {
		branches := make([]managementapi.AuthenticationRequirement, len(requirement.AnyOf))
		for index, branch := range requirement.AnyOf {
			wire := managementapi.AuthenticationRequirement{Kind: string(branch.Kind)}
			if branch.Human != nil {
				wire.Human = &managementapi.HumanRequirement{MinimumAAL: branch.Human.MinimumAAL, AcceptedAMR: slices.Clone(branch.Human.AcceptedAMR), MaxAuthenticationAgeSeconds: branch.Human.MaxAuthenticationAgeSeconds}
			}
			if branch.Workload != nil {
				wire.Workload = &managementapi.WorkloadRequirement{MinimumWorkloadClass: branch.Workload.MinimumWorkloadClass, MaxSourceAgeSeconds: branch.Workload.MaxSourceAgeSeconds}
			}
			branches[index] = wire
		}
		result[action] = branches
	}
	return result
}

func requirementsFromDTO(source map[string][]managementapi.AuthenticationRequirement) map[string]managementauth.ActionRequirement {
	result := make(map[string]managementauth.ActionRequirement, len(source))
	for action, values := range source {
		requirement := managementauth.ActionRequirement{AnyOf: make([]managementauth.AuthenticationRequirement, len(values))}
		for index, value := range values {
			branch := managementauth.AuthenticationRequirement{Kind: managementauth.AuthenticationRequirementKind(value.Kind)}
			if value.Human != nil {
				branch.Human = &managementauth.HumanRequirement{MinimumAAL: value.Human.MinimumAAL, AcceptedAMR: slices.Clone(value.Human.AcceptedAMR), MaxAuthenticationAgeSeconds: value.Human.MaxAuthenticationAgeSeconds}
			}
			if value.Workload != nil {
				branch.Workload = &managementauth.WorkloadRequirement{MinimumWorkloadClass: value.Workload.MinimumWorkloadClass, MaxSourceAgeSeconds: value.Workload.MaxSourceAgeSeconds}
			}
			requirement.AnyOf[index] = branch
		}
		result[action] = requirement
	}
	return result
}

func routingClaimSchemaDTO(value namespacemanagement.RoutingClaimSchema) managementapi.RoutingClaimSchema {
	definitions := make(map[string]managementapi.RoutingClaimDefinition, len(value.Definitions))
	for name, definition := range value.Definitions {
		definitions[name] = managementapi.RoutingClaimDefinition{Kind: definition.Kind, Minimum: definition.Minimum, Maximum: definition.Maximum, MaxLength: definition.MaxLength}
	}
	return managementapi.RoutingClaimSchema{NamespaceID: value.NamespaceID, Definitions: definitions, Revision: value.Revision, UpdatedAt: value.UpdatedAt}
}

func routingClaimDefinitionsFromDTO(source map[string]managementapi.RoutingClaimDefinition) map[string]accessmanagement.ClaimDefinition {
	result := make(map[string]accessmanagement.ClaimDefinition, len(source))
	for name, definition := range source {
		result[name] = accessmanagement.ClaimDefinition{Kind: definition.Kind, Minimum: definition.Minimum, Maximum: definition.Maximum, MaxLength: definition.MaxLength}
	}
	return result
}

func capabilitiesFromDTO(source []string) []accesscontrol.TeamAdminCapability {
	result := make([]accesscontrol.TeamAdminCapability, len(source))
	for index := range source {
		result[index] = accesscontrol.TeamAdminCapability(source[index])
	}
	return result
}
