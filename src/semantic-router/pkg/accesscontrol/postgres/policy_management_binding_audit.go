package postgres

import "github.com/vllm-project/semantic-router/src/semantic-router/pkg/policymanagement"

func managedBindingReferences(policyID string, subject policymanagement.Subject) map[string]string {
	return map[string]string{
		"policyId": policyID, "subjectId": subject.ID, "subjectType": string(subject.Type),
	}
}

func managedRateBindingReferences(binding policymanagement.RateLimitBinding) map[string]string {
	references := managedBindingReferences(binding.PolicyID, binding.Subject)
	references["bindingMode"] = string(binding.Mode)
	references["quotaPartitionId"] = binding.QuotaPartitionID
	return references
}

func managedBindingAuditDetails(policyID string, subject policymanagement.Subject) map[string]string {
	return managedBindingReferences(policyID, subject)
}
