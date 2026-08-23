package managementapi

import "time"

// PolicySubject is the public typed subject reference shared by both binding
// families. Namespace ownership is derived from the authenticated request and
// is never accepted from the body.
type PolicySubject struct {
	Type string `json:"type"`
	ID   string `json:"id"`
}

type AccessPolicyGrant struct {
	ResourceType string `json:"resourceType"`
	ResourceID   string `json:"resourceId"`
	Permission   string `json:"permission"`
	Effect       string `json:"effect"`
}

type AccessPolicyCreateRequest struct {
	Name        string              `json:"name"`
	Description string              `json:"description,omitempty"`
	Status      string              `json:"status,omitempty"`
	Grants      []AccessPolicyGrant `json:"grants,omitempty"`
}

type AccessPolicyPatchRequest struct {
	Name        *string              `json:"name,omitempty"`
	Description *string              `json:"description,omitempty"`
	Status      *string              `json:"status,omitempty"`
	Grants      *[]AccessPolicyGrant `json:"grants,omitempty"`
}

type AccessPolicy struct {
	PolicyID    string              `json:"policyId"`
	Name        string              `json:"name"`
	Description string              `json:"description"`
	Status      string              `json:"status"`
	Revision    uint64              `json:"revision"`
	Grants      []AccessPolicyGrant `json:"grants"`
	CreatedAt   time.Time           `json:"createdAt"`
	UpdatedAt   time.Time           `json:"updatedAt"`
}

type AccessPolicyPage struct {
	Data []AccessPolicy `json:"data"`
	Page PageInfo       `json:"page"`
}

type AccessPolicyDetail struct {
	Data AccessPolicy `json:"data"`
}

// RateLimitRuleInput omits Ordinal, which is assigned canonically by the
// policy compiler. RuleID is optional for new rules and required to preserve a
// counter across a limit-only edit.
type RateLimitRuleInput struct {
	RuleID           string `json:"ruleId,omitempty"`
	Metric           string `json:"metric"`
	Algorithm        string `json:"algorithm"`
	Limit            string `json:"limit,omitempty"`
	Window           string `json:"window,omitempty"`
	Period           string `json:"period,omitempty"`
	Timezone         string `json:"timezone,omitempty"`
	Capacity         string `json:"capacity,omitempty"`
	RefillAmount     string `json:"refillAmount,omitempty"`
	RefillPeriod     string `json:"refillPeriod,omitempty"`
	EmissionInterval string `json:"emissionInterval,omitempty"`
	BurstTolerance   *int64 `json:"burstTolerance,omitempty"`
	Accounting       string `json:"accounting"`
	Enforcement      string `json:"enforcement"`
}

type RateLimitRule struct {
	RateLimitRuleInput
	Ordinal uint32 `json:"ordinal"`
}

type RateLimitPolicyCreateRequest struct {
	Name        string               `json:"name"`
	Description string               `json:"description,omitempty"`
	Status      string               `json:"status,omitempty"`
	Rules       []RateLimitRuleInput `json:"rules,omitempty"`
}

type RateLimitPolicyPatchRequest struct {
	Name        *string               `json:"name,omitempty"`
	Description *string               `json:"description,omitempty"`
	Status      *string               `json:"status,omitempty"`
	Rules       *[]RateLimitRuleInput `json:"rules,omitempty"`
}

type RateLimitPolicy struct {
	PolicyID    string          `json:"policyId"`
	Name        string          `json:"name"`
	Description string          `json:"description"`
	Status      string          `json:"status"`
	Revision    uint64          `json:"revision"`
	Rules       []RateLimitRule `json:"rules"`
	CreatedAt   time.Time       `json:"createdAt"`
	UpdatedAt   time.Time       `json:"updatedAt"`
}

type RateLimitPolicyPage struct {
	Data []RateLimitPolicy `json:"data"`
	Page PageInfo          `json:"page"`
}

type RateLimitPolicyDetail struct {
	Data RateLimitPolicy `json:"data"`
}

type AccessPolicyBindingCreateRequest struct {
	PolicyID string        `json:"policyId"`
	Subject  PolicySubject `json:"subject"`
}

type AccessPolicyBindingPatchRequest struct {
	Status string `json:"status"`
}

type AccessPolicyBindingBulkApplyRequest struct {
	Items []AccessPolicyBindingBulkApplyItem `json:"items"`
}

type AccessPolicyBindingBulkApplyItem struct {
	ItemID   string        `json:"itemId"`
	PolicyID string        `json:"policyId"`
	Subject  PolicySubject `json:"subject"`
}

type AccessPolicyBinding struct {
	BindingID string        `json:"bindingId"`
	PolicyID  string        `json:"policyId"`
	Subject   PolicySubject `json:"subject"`
	Status    string        `json:"status"`
	Revision  uint64        `json:"revision"`
	CreatedAt time.Time     `json:"createdAt"`
	UpdatedAt time.Time     `json:"updatedAt"`
}

type AccessPolicyBindingPage struct {
	Data []AccessPolicyBinding `json:"data"`
	Page PageInfo              `json:"page"`
}

type AccessPolicyBindingDetail struct {
	Data AccessPolicyBinding `json:"data"`
}

type InlineRateLimitPolicy struct {
	Name        string               `json:"name"`
	Description string               `json:"description,omitempty"`
	Rules       []RateLimitRuleInput `json:"rules"`
}

// RateLimitBindingCreateRequest is a strict one-of at the transport boundary:
// PolicyID selects an existing policy, while InlinePolicy materializes an
// ordinary reusable policy in the same transaction as its binding.
type RateLimitBindingCreateRequest struct {
	PolicyID     string                 `json:"policyId,omitempty"`
	InlinePolicy *InlineRateLimitPolicy `json:"inlinePolicy,omitempty"`
	Subject      PolicySubject          `json:"subject"`
	Mode         string                 `json:"mode"`
}

type RateLimitBindingPatchRequest struct {
	Status string `json:"status"`
}

type RateLimitBindingBulkApplyRequest struct {
	Items []RateLimitBindingBulkApplyItem `json:"items"`
}

type RateLimitBindingBulkApplyItem struct {
	ItemID       string                 `json:"itemId"`
	PolicyID     string                 `json:"policyId,omitempty"`
	InlinePolicy *InlineRateLimitPolicy `json:"inlinePolicy,omitempty"`
	Subject      PolicySubject          `json:"subject"`
	Mode         string                 `json:"mode"`
}

type RateLimitBinding struct {
	BindingID        string        `json:"bindingId"`
	PolicyID         string        `json:"policyId"`
	Subject          PolicySubject `json:"subject"`
	Mode             string        `json:"mode"`
	QuotaPartitionID string        `json:"quotaPartitionId"`
	Status           string        `json:"status"`
	Revision         uint64        `json:"revision"`
	CreatedAt        time.Time     `json:"createdAt"`
	UpdatedAt        time.Time     `json:"updatedAt"`
}

type RateLimitBindingPage struct {
	Data []RateLimitBinding `json:"data"`
	Page PageInfo           `json:"page"`
}

type RateLimitBindingDetail struct {
	Data RateLimitBinding `json:"data"`
}

type RateLimitBindingCreateReceipt struct {
	BindingID     string               `json:"bindingId"`
	PolicyID      string               `json:"policyId"`
	Revision      uint64               `json:"revision"`
	CreatedPolicy bool                 `json:"createdPolicy"`
	Idempotency   *IdempotencyMetadata `json:"idempotency,omitempty"`
}
