package policymanagement

import (
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
)

// InlineRateLimitPolicySpec contains every deterministic input needed to turn
// API-key inline rules into an ordinary reusable RateLimitPolicy. NewRuleID is
// called only for rules whose ID is empty; caller-supplied IDs are preserved so
// an intentional limit-only edit can retain counter identity.
type InlineRateLimitPolicySpec struct {
	NamespaceID string
	PolicyID    string
	Name        string
	Description string
	Rules       []RateLimitRule
	Now         time.Time
	NewRuleID   func() string
}

// CompileInlineRateLimitPolicy is pure with respect to persistence. For fixed
// inputs and ID-source results it returns the same canonical active policy and
// never mutates the caller's rule slice.
func CompileInlineRateLimitPolicy(spec InlineRateLimitPolicySpec) (RateLimitPolicy, error) {
	name := strings.TrimSpace(spec.Name)
	description := strings.TrimSpace(spec.Description)
	if !canonicalUUID(spec.NamespaceID) || !canonicalUUID(spec.PolicyID) || spec.Now.IsZero() ||
		validatePolicyMetadata(name, description, accesscontrol.PolicyStatusActive) != nil ||
		len(spec.Rules) == 0 || len(spec.Rules) > maximumRules {
		return RateLimitPolicy{}, ErrInvalidRequest
	}

	rules, err := compileRateLimitRules(spec.PolicyID, spec.Rules, spec.NewRuleID)
	if err != nil {
		return RateLimitPolicy{}, err
	}
	now := spec.Now.UTC().Truncate(time.Microsecond)
	policy := RateLimitPolicy{
		ID:          spec.PolicyID,
		NamespaceID: spec.NamespaceID,
		Name:        name,
		Description: description,
		Status:      accesscontrol.PolicyStatusActive,
		Revision:    1,
		Rules:       rules,
		CreatedAt:   now,
		UpdatedAt:   now,
	}
	if validateRules(policy.ID, policy.Rules) != nil {
		return RateLimitPolicy{}, ErrInvalidRequest
	}
	return policy, nil
}

func compileRateLimitRules(
	policyID string,
	input []RateLimitRule,
	newRuleID func() string,
) ([]RateLimitRule, error) {
	if !canonicalUUID(policyID) || len(input) > maximumRules {
		return nil, ErrInvalidRequest
	}
	rules := cloneRules(input)
	for index := range rules {
		if rules[index].ID == "" {
			if newRuleID == nil {
				return nil, ErrInvalidRequest
			}
			rules[index].ID = newRuleID()
		}
		rules[index].Ordinal = uint32(index)
	}
	if validateRules(policyID, rules) != nil {
		return nil, ErrInvalidRequest
	}
	return rules, nil
}
