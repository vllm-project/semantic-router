package managementserver

import (
	"encoding/json"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementapi"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/policymanagement"
)

func newAccessPolicy(value policymanagement.AccessPolicy) managementapi.AccessPolicy {
	grants := make([]managementapi.AccessPolicyGrant, len(value.Grants))
	for index, grant := range value.Grants {
		grants[index] = managementapi.AccessPolicyGrant{
			ResourceType: string(grant.ResourceType), ResourceID: grant.ResourceID,
			Permission: string(grant.Permission), Effect: string(grant.Effect),
		}
	}
	return managementapi.AccessPolicy{
		PolicyID: value.ID, Name: value.Name, Description: value.Description,
		Status: string(value.Status), Revision: value.Revision, Grants: grants,
		CreatedAt: value.CreatedAt, UpdatedAt: value.UpdatedAt,
	}
}

func newAccessPolicyPage(value policymanagement.Page[policymanagement.AccessPolicy]) managementapi.AccessPolicyPage {
	items := make([]managementapi.AccessPolicy, len(value.Items))
	for index := range value.Items {
		items[index] = newAccessPolicy(value.Items[index])
	}
	return managementapi.AccessPolicyPage{Data: items, Page: managementapi.PageInfo{
		NextCursor: value.NextCursor, HasMore: value.HasMore, PageSize: value.PageSize,
	}}
}

func policyGrants(input []managementapi.AccessPolicyGrant) []policymanagement.AccessGrant {
	result := make([]policymanagement.AccessGrant, len(input))
	for index, grant := range input {
		result[index] = policymanagement.AccessGrant{
			ResourceType: accesscontrol.GrantResourceType(grant.ResourceType), ResourceID: grant.ResourceID,
			Permission: accesscontrol.GrantPermission(grant.Permission), Effect: accesscontrol.GrantEffect(grant.Effect),
		}
	}
	return result
}

func newRateLimitPolicy(value policymanagement.RateLimitPolicy) managementapi.RateLimitPolicy {
	rules := make([]managementapi.RateLimitRule, len(value.Rules))
	for index := range value.Rules {
		rules[index] = newRateLimitRule(value.Rules[index])
	}
	return managementapi.RateLimitPolicy{
		PolicyID: value.ID, Name: value.Name, Description: value.Description,
		Status: string(value.Status), Revision: value.Revision, Rules: rules,
		CreatedAt: value.CreatedAt, UpdatedAt: value.UpdatedAt,
	}
}

func newRateLimitRule(value policymanagement.RateLimitRule) managementapi.RateLimitRule {
	return managementapi.RateLimitRule{RateLimitRuleInput: managementapi.RateLimitRuleInput{
		RuleID: value.ID, Metric: string(value.Metric), Algorithm: string(value.Algorithm),
		Limit: string(value.Limit), Window: policyDurationString(value.Window),
		Period: string(value.CalendarPeriod), Timezone: value.Timezone,
		Capacity: string(value.BucketCapacity), RefillAmount: string(value.RefillAmount),
		RefillPeriod:     policyDurationString(value.RefillPeriod),
		EmissionInterval: policyDurationString(value.GCRAEmissionInterval),
		BurstTolerance:   clonePolicyBurstTolerance(value.GCRABurstTolerance),
		Accounting:       string(value.Accounting), Enforcement: string(value.Enforcement),
	}, Ordinal: value.Ordinal}
}

func newRateLimitPolicyPage(value policymanagement.Page[policymanagement.RateLimitPolicy]) managementapi.RateLimitPolicyPage {
	items := make([]managementapi.RateLimitPolicy, len(value.Items))
	for index := range value.Items {
		items[index] = newRateLimitPolicy(value.Items[index])
	}
	return managementapi.RateLimitPolicyPage{Data: items, Page: managementapi.PageInfo{
		NextCursor: value.NextCursor, HasMore: value.HasMore, PageSize: value.PageSize,
	}}
}

func policyRules(input []managementapi.RateLimitRuleInput) ([]policymanagement.RateLimitRule, error) {
	result := make([]policymanagement.RateLimitRule, len(input))
	for index, rule := range input {
		window, err := parsePolicyDuration(rule.Window)
		if err != nil {
			return nil, err
		}
		refillPeriod, err := parsePolicyDuration(rule.RefillPeriod)
		if err != nil {
			return nil, err
		}
		emissionInterval, err := parsePolicyDuration(rule.EmissionInterval)
		if err != nil {
			return nil, err
		}
		result[index] = policymanagement.RateLimitRule{
			ID: rule.RuleID, Metric: accesscontrol.RateMetric(rule.Metric),
			Algorithm: accesscontrol.RateAlgorithm(rule.Algorithm), Limit: accesscontrol.QuotaValue(rule.Limit),
			Window: window, CalendarPeriod: accesscontrol.CalendarPeriod(rule.Period), Timezone: rule.Timezone,
			BucketCapacity: accesscontrol.QuotaValue(rule.Capacity), RefillAmount: accesscontrol.QuotaValue(rule.RefillAmount),
			RefillPeriod: refillPeriod, GCRAEmissionInterval: emissionInterval,
			GCRABurstTolerance: clonePolicyBurstTolerance(rule.BurstTolerance),
			Accounting:         accesscontrol.RateAccounting(rule.Accounting),
			Enforcement:        accesscontrol.RateEnforcement(rule.Enforcement),
		}
	}
	return result, nil
}

func parsePolicyDuration(value string) (policymanagement.ISODuration, error) {
	if value == "" {
		return 0, nil
	}
	encoded, err := json.Marshal(value)
	if err != nil {
		return 0, policymanagement.ErrInvalidRequest
	}
	var duration policymanagement.ISODuration
	if err := json.Unmarshal(encoded, &duration); err != nil {
		return 0, err
	}
	return duration, nil
}

func policyDurationString(value policymanagement.ISODuration) string {
	if value.Duration() == 0 {
		return ""
	}
	return value.String()
}

func policyStatusPointer(value *string) *accesscontrol.PolicyStatus {
	if value == nil {
		return nil
	}
	status := accesscontrol.PolicyStatus(*value)
	return &status
}

func newAccessPolicyBinding(value policymanagement.AccessPolicyBinding) managementapi.AccessPolicyBinding {
	return managementapi.AccessPolicyBinding{
		BindingID: value.ID, PolicyID: value.PolicyID,
		Subject: managementapi.PolicySubject{Type: string(value.Subject.Type), ID: value.Subject.ID},
		Status:  string(value.Status), Revision: value.Revision,
		CreatedAt: value.CreatedAt, UpdatedAt: value.UpdatedAt,
	}
}

func newAccessPolicyBindingPage(value policymanagement.Page[policymanagement.AccessPolicyBinding]) managementapi.AccessPolicyBindingPage {
	items := make([]managementapi.AccessPolicyBinding, len(value.Items))
	for index := range value.Items {
		items[index] = newAccessPolicyBinding(value.Items[index])
	}
	return managementapi.AccessPolicyBindingPage{Data: items, Page: managementapi.PageInfo{
		NextCursor: value.NextCursor, HasMore: value.HasMore, PageSize: value.PageSize,
		TotalCount: pageTotalCount(value.TotalCount),
	}}
}

func newRateLimitBinding(value policymanagement.RateLimitBinding) managementapi.RateLimitBinding {
	return managementapi.RateLimitBinding{
		BindingID: value.ID, PolicyID: value.PolicyID,
		Subject: managementapi.PolicySubject{Type: string(value.Subject.Type), ID: value.Subject.ID},
		Mode:    string(value.Mode), QuotaPartitionID: value.QuotaPartitionID,
		Status: string(value.Status), Revision: value.Revision,
		CreatedAt: value.CreatedAt, UpdatedAt: value.UpdatedAt,
	}
}

func newRateLimitBindingPage(value policymanagement.Page[policymanagement.RateLimitBinding]) managementapi.RateLimitBindingPage {
	items := make([]managementapi.RateLimitBinding, len(value.Items))
	for index := range value.Items {
		items[index] = newRateLimitBinding(value.Items[index])
	}
	return managementapi.RateLimitBindingPage{Data: items, Page: managementapi.PageInfo{
		NextCursor: value.NextCursor, HasMore: value.HasMore, PageSize: value.PageSize,
		TotalCount: pageTotalCount(value.TotalCount),
	}}
}

func clonePolicyBurstTolerance(value *int64) *int64 {
	if value == nil {
		return nil
	}
	cloned := *value
	return &cloned
}
