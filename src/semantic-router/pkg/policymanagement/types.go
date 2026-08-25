package policymanagement

import (
	"net/netip"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementcommand"
)

type Actor struct {
	PrincipalID string
	ActorChain  []string
	RequestID   string
	SourceIP    netip.Addr
}

type AccessGrant struct {
	ResourceType accesscontrol.GrantResourceType `json:"resourceType"`
	ResourceID   string                          `json:"resourceId"`
	Permission   accesscontrol.GrantPermission   `json:"permission"`
	Effect       accesscontrol.GrantEffect       `json:"effect"`
}

type AccessPolicy struct {
	ID          string                     `json:"policyId"`
	NamespaceID string                     `json:"namespaceId"`
	Name        string                     `json:"name"`
	Description string                     `json:"description"`
	Status      accesscontrol.PolicyStatus `json:"status"`
	Revision    uint64                     `json:"revision"`
	Grants      []AccessGrant              `json:"grants"`
	CreatedAt   time.Time                  `json:"createdAt"`
	UpdatedAt   time.Time                  `json:"updatedAt"`
}

// RateLimitRule is the public discriminated-union value. ID is optional on
// create/update input and always populated on reads. List order is canonical;
// Ordinal is output-only and mirrors that order.
type RateLimitRule struct {
	ID                   string                        `json:"ruleId,omitempty"`
	Metric               accesscontrol.RateMetric      `json:"metric"`
	Algorithm            accesscontrol.RateAlgorithm   `json:"algorithm"`
	Limit                accesscontrol.QuotaValue      `json:"limit,omitempty"`
	Window               ISODuration                   `json:"window,omitempty"`
	CalendarPeriod       accesscontrol.CalendarPeriod  `json:"period,omitempty"`
	Timezone             string                        `json:"timezone,omitempty"`
	BucketCapacity       accesscontrol.QuotaValue      `json:"capacity,omitempty"`
	RefillAmount         accesscontrol.QuotaValue      `json:"refillAmount,omitempty"`
	RefillPeriod         ISODuration                   `json:"refillPeriod,omitempty"`
	GCRAEmissionInterval ISODuration                   `json:"emissionInterval,omitempty"`
	GCRABurstTolerance   *int64                        `json:"burstTolerance,omitempty"`
	Accounting           accesscontrol.RateAccounting  `json:"accounting"`
	Enforcement          accesscontrol.RateEnforcement `json:"enforcement"`
	Ordinal              uint32                        `json:"ordinal"`
}

type RateLimitPolicy struct {
	ID          string                     `json:"policyId"`
	NamespaceID string                     `json:"namespaceId"`
	Name        string                     `json:"name"`
	Description string                     `json:"description"`
	Status      accesscontrol.PolicyStatus `json:"status"`
	Revision    uint64                     `json:"revision"`
	Rules       []RateLimitRule            `json:"rules"`
	CreatedAt   time.Time                  `json:"createdAt"`
	UpdatedAt   time.Time                  `json:"updatedAt"`
}

type Subject struct {
	Type accesscontrol.SubjectKind `json:"type"`
	ID   string                    `json:"id"`
}

type AccessPolicyBinding struct {
	ID          string                      `json:"bindingId"`
	NamespaceID string                      `json:"namespaceId"`
	PolicyID    string                      `json:"policyId"`
	Subject     Subject                     `json:"subject"`
	Status      accesscontrol.BindingStatus `json:"status"`
	Revision    uint64                      `json:"revision"`
	CreatedAt   time.Time                   `json:"createdAt"`
	UpdatedAt   time.Time                   `json:"updatedAt"`
}

type RateLimitBinding struct {
	ID               string                        `json:"bindingId"`
	NamespaceID      string                        `json:"namespaceId"`
	PolicyID         string                        `json:"policyId"`
	Subject          Subject                       `json:"subject"`
	Mode             accesscontrol.RateBindingMode `json:"mode"`
	QuotaPartitionID string                        `json:"quotaPartitionId"`
	Status           accesscontrol.BindingStatus   `json:"status"`
	Revision         uint64                        `json:"revision"`
	CreatedAt        time.Time                     `json:"createdAt"`
	UpdatedAt        time.Time                     `json:"updatedAt"`
}

type MutationResult struct {
	Kind       string
	ID         string
	Revision   uint64
	Replayed   bool
	HTTPStatus int
}

type InlineRateBindingResult struct {
	Policy  RateLimitPolicy
	Binding RateLimitBinding
	Created bool
	MutationResult
}

type Page[T any] struct {
	Items      []T
	NextCursor string
	HasMore    bool
	PageSize   int
	TotalCount *uint64
}

type ListPoliciesRequest struct {
	NamespaceID string
	Status      accesscontrol.PolicyStatus
	Search      string
	Cursor      string
	PageSize    int
	Scope       accesscontrol.ResultScope
}

type ListBindingsRequest struct {
	NamespaceID  string
	PolicyID     string
	Subject      *Subject
	Status       accesscontrol.BindingStatus
	Mode         accesscontrol.RateBindingMode
	Cursor       string
	PageSize     int
	IncludeTotal bool
	Scope        accesscontrol.ResultScope
}

type CreateAccessPolicyRequest struct {
	NamespaceID    string
	Name           string
	Description    string
	Status         accesscontrol.PolicyStatus
	Grants         []AccessGrant
	IdempotencyKey string
	Actor          Actor
}

type UpdateAccessPolicyRequest struct {
	NamespaceID      string
	PolicyID         string
	ExpectedRevision uint64
	Name             *string
	Description      *string
	Status           *accesscontrol.PolicyStatus
	Grants           *[]AccessGrant
	Actor            Actor
}

type CreateRateLimitPolicyRequest struct {
	NamespaceID    string
	Name           string
	Description    string
	Status         accesscontrol.PolicyStatus
	Rules          []RateLimitRule
	IdempotencyKey string
	Actor          Actor
}

type UpdateRateLimitPolicyRequest struct {
	NamespaceID      string
	PolicyID         string
	ExpectedRevision uint64
	Name             *string
	Description      *string
	Status           *accesscontrol.PolicyStatus
	Rules            *[]RateLimitRule
	Actor            Actor
}

type DeletePolicyRequest struct {
	NamespaceID      string
	PolicyID         string
	ExpectedRevision uint64
	Actor            Actor
}

type CreateAccessBindingRequest struct {
	NamespaceID    string
	PolicyID       string
	Subject        Subject
	IdempotencyKey string
	Actor          Actor
}

type CreateRateBindingRequest struct {
	NamespaceID    string
	PolicyID       string
	Subject        Subject
	Mode           accesscontrol.RateBindingMode
	IdempotencyKey string
	Actor          Actor
}

type CreateInlineRateBindingRequest struct {
	NamespaceID    string
	Name           string
	Description    string
	Rules          []RateLimitRule
	Subject        Subject
	Mode           accesscontrol.RateBindingMode
	IdempotencyKey string
	Actor          Actor
}

type UpdateBindingRequest struct {
	NamespaceID      string
	BindingID        string
	ExpectedRevision uint64
	Status           accesscontrol.BindingStatus
	Actor            Actor
}

type DeleteBindingRequest struct {
	NamespaceID      string
	BindingID        string
	ExpectedRevision uint64
	Actor            Actor
}

type CreateAccessPolicyMutation struct {
	Policy  AccessPolicy
	Command managementcommand.Command
	Actor   Actor
}

type CreateRateLimitPolicyMutation struct {
	Policy  RateLimitPolicy
	Command managementcommand.Command
	Actor   Actor
}

type CreateAccessBindingMutation struct {
	Binding AccessPolicyBinding
	Command managementcommand.Command
	Actor   Actor
}

type CreateRateBindingMutation struct {
	Binding RateLimitBinding
	Command managementcommand.Command
	Actor   Actor
}

type CreateInlineRateBindingMutation struct {
	Policy  RateLimitPolicy
	Binding RateLimitBinding
	Command managementcommand.Command
	Actor   Actor
}
