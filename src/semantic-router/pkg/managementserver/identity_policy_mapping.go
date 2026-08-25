package managementserver

import (
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementapi"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauth"
)

func policyDTO(policy managementauth.SessionPolicy) managementapi.ManagementSessionPolicy {
	requirements := map[string][]managementapi.AuthenticationRequirement{}
	for action, requirement := range policy.ActionRequirements {
		values := make([]managementapi.AuthenticationRequirement, len(requirement.AnyOf))
		for index, branch := range requirement.AnyOf {
			value := managementapi.AuthenticationRequirement{Kind: string(branch.Kind)}
			if branch.Human != nil {
				value.Human = &managementapi.HumanRequirement{
					MinimumAAL: branch.Human.MinimumAAL, AcceptedAMR: branch.Human.AcceptedAMR,
					MaxAuthenticationAgeSeconds: branch.Human.MaxAuthenticationAgeSeconds,
				}
			}
			if branch.Workload != nil {
				value.Workload = &managementapi.WorkloadRequirement{
					MinimumWorkloadClass: branch.Workload.MinimumWorkloadClass,
					MaxSourceAgeSeconds:  branch.Workload.MaxSourceAgeSeconds,
				}
			}
			values[index] = value
		}
		requirements[action] = values
	}
	return managementapi.ManagementSessionPolicy{
		AccessTokenTTLSeconds: int64(policy.AccessTokenTTL / time.Second),
		SessionTTLSeconds:     int64(policy.SessionTTL / time.Second),
		MaxActiveSessions:     policy.MaxActiveSessions, ActionRequirements: requirements,
		SeedVersion: policy.SeedVersion, Revision: policy.Revision, UpdatedAt: policy.UpdatedAt,
	}
}

func policyFromDTO(body managementapi.ManagementSessionPolicyPatchRequest, revision uint64, now time.Time) (managementauth.SessionPolicy, error) {
	requirements := map[string]managementauth.ActionRequirement{}
	for action, values := range body.ActionRequirements {
		target := managementauth.ActionRequirement{AnyOf: make([]managementauth.AuthenticationRequirement, len(values))}
		for index, value := range values {
			branch := managementauth.AuthenticationRequirement{Kind: managementauth.AuthenticationRequirementKind(value.Kind)}
			if value.Human != nil {
				branch.Human = &managementauth.HumanRequirement{
					MinimumAAL: value.Human.MinimumAAL, AcceptedAMR: value.Human.AcceptedAMR,
					MaxAuthenticationAgeSeconds: value.Human.MaxAuthenticationAgeSeconds,
				}
			}
			if value.Workload != nil {
				branch.Workload = &managementauth.WorkloadRequirement{
					MinimumWorkloadClass: value.Workload.MinimumWorkloadClass,
					MaxSourceAgeSeconds:  value.Workload.MaxSourceAgeSeconds,
				}
			}
			target.AnyOf[index] = branch
		}
		requirements[action] = target
	}
	policy := managementauth.SessionPolicy{
		AccessTokenTTL:    time.Duration(body.AccessTokenTTLSeconds) * time.Second,
		SessionTTL:        time.Duration(body.SessionTTLSeconds) * time.Second,
		MaxActiveSessions: body.MaxActiveSessions, ActionRequirements: requirements,
		SeedVersion: managementauth.SupportedSessionPolicySeedVersion, Revision: revision, UpdatedAt: now,
	}
	return policy, policy.Validate()
}
