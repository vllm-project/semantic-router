// Package accessprojection compiles authoritative access-control resources
// into deterministic, immutable values applied to the shared runtime store.
package accessprojection

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"sort"
	"strings"
	"time"
	_ "time/tzdata"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/quota"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routingsnapshot"
)

// Candidate contains one complete key projection. Store adapters load these
// resources in a repeatable-read transaction; the compiler performs no joins.
type Candidate struct {
	Revision      uint64
	Namespace     accesscontrol.Namespace
	Key           accesscontrol.APIKey
	Relationships accesscontrol.APIKeyRelationships

	KeyAccessBindings  []accesscontrol.AccessPolicyBinding
	UserAccessBindings []accesscontrol.AccessPolicyBinding
	TeamAccessBindings []accesscontrol.AccessPolicyBinding
	AccessPolicies     map[accesscontrol.AccessPolicyID]accesscontrol.AccessPolicy

	KeyRateBindings  []accesscontrol.RateLimitBinding
	UserRateBindings []accesscontrol.RateLimitBinding
	TeamRateBindings []accesscontrol.RateLimitBinding
	RatePolicies     map[accesscontrol.RateLimitPolicyID]accesscontrol.RateLimitPolicy

	RoutingClaims map[string]routingsnapshot.ClaimValue
}

// CompileOptions supplies deterministic publication facts that cannot be
// inferred from desired-state rows. CalendarScheduleStart must be UTC and
// millisecond-aligned when any calendar-window rule is reachable.
type CompileOptions struct {
	CalendarScheduleStart time.Time
}

// EmbeddedTZDBVersion is part of the Router binary capability, not mutable
// user configuration. All replicas compile calendar windows from the same
// embedded IANA database and publish this identifier with every schedule.
const EmbeddedTZDBVersion = "iana-2025b"

// Projection is the single access:policy:<key>:<revision> runtime value.
type Projection struct {
	NamespaceID     string                                `json:"namespaceId"`
	QuotaPartition  string                                `json:"quotaPartition"`
	BillingCurrency string                                `json:"billingCurrency"`
	KeyID           string                                `json:"keyId"`
	KeyStatus       accesscontrol.APIKeyStatus            `json:"keyStatus"`
	KeyExpiresAt    *time.Time                            `json:"keyExpiresAt,omitempty"`
	UserID          string                                `json:"userId,omitempty"`
	TeamID          string                                `json:"teamId,omitempty"`
	Revision        uint64                                `json:"revision"`
	PolicyEpoch     uint64                                `json:"policyEpoch"`
	DelegationEpoch uint64                                `json:"delegationEpoch"`
	AccessSource    accesscontrol.InheritanceLayer        `json:"accessSource"`
	Grants          []Grant                               `json:"grants"`
	RateBindings    []RateBinding                         `json:"rateBindings"`
	RoutingClaims   map[string]routingsnapshot.ClaimValue `json:"routingClaims,omitempty"`
	Digest          string                                `json:"digest"`
}

type Grant struct {
	BindingID    string                          `json:"bindingId"`
	PolicyID     string                          `json:"policyId"`
	Source       accesscontrol.InheritanceLayer  `json:"source"`
	ResourceType accesscontrol.GrantResourceType `json:"resourceType"`
	ResourceID   string                          `json:"resourceId"`
	Permission   accesscontrol.GrantPermission   `json:"permission"`
	Effect       accesscontrol.GrantEffect       `json:"effect"`
}

type RateBinding struct {
	BindingID string                         `json:"bindingId"`
	PolicyID  string                         `json:"policyId"`
	SubjectID string                         `json:"subjectId"`
	Source    accesscontrol.InheritanceLayer `json:"source"`
	Mode      accesscontrol.RateBindingMode  `json:"mode"`
	Rules     []ProjectedRateRule            `json:"rules"`
}

type ProjectedRateRule struct {
	Rule             quota.RateLimitRule `json:"rule"`
	TZDBVersion      string              `json:"tzdbVersion,omitempty"`
	CalendarSchedule []CalendarInterval  `json:"calendarSchedule,omitempty"`
}

type CalendarInterval struct {
	Start time.Time `json:"start"`
	End   time.Time `json:"end"`
}

// CredentialProjection is the O(1) public-kid lookup value. Reveal material is
// intentionally impossible to represent here.
type CredentialProjection struct {
	Kind          string     `json:"kind"`
	KID           string     `json:"kid"`
	KeyID         string     `json:"keyId"`
	SecretHMAC    []byte     `json:"secretHmac"`
	PepperVersion string     `json:"pepperVersion"`
	Status        string     `json:"status"`
	NotBefore     time.Time  `json:"notBefore"`
	ExpiresAt     *time.Time `json:"expiresAt,omitempty"`

	ManagementSessionID string `json:"managementSessionId,omitempty"`
	PrincipalID         string `json:"principalId,omitempty"`
	DelegationEpoch     uint64 `json:"delegationEpoch,omitempty"`
	UserID              string `json:"userId,omitempty"`
	TeamID              string `json:"teamId,omitempty"`
	Audience            string `json:"audience,omitempty"`
}

// DelegationContext is the complete authority binding of one delegated
// inference credential. It is immutable for the lifetime of the credential.
type DelegationContext struct {
	ManagementSessionID string
	PrincipalID         string
	DelegationEpoch     uint64
	UserID              string
	TeamID              string
	Audience            string
}

func CompileCredential(
	kind string,
	credential accesscontrol.CredentialVersion,
	delegation *DelegationContext,
) (CredentialProjection, error) {
	if err := credential.Validate(); err != nil {
		return CredentialProjection{}, err
	}
	if credential.Status != accesscontrol.CredentialStatusActive &&
		credential.Status != accesscontrol.CredentialStatusRetiring {
		return CredentialProjection{}, fmt.Errorf("credential %s is not publishable", credential.ID)
	}
	if kind != "api_key" && kind != "delegation" {
		return CredentialProjection{}, fmt.Errorf("credential %s has an unsupported kind", credential.ID)
	}
	if kind == "delegation" {
		if delegation == nil || strings.TrimSpace(delegation.ManagementSessionID) == "" ||
			strings.TrimSpace(delegation.PrincipalID) == "" || delegation.DelegationEpoch == 0 ||
			strings.TrimSpace(delegation.UserID) == "" || strings.TrimSpace(delegation.Audience) == "" {
			return CredentialProjection{}, fmt.Errorf("credential %s has incomplete delegation context", credential.ID)
		}
	} else if delegation != nil {
		return CredentialProjection{}, fmt.Errorf("API-key credential %s cannot carry delegation context", credential.ID)
	}
	projection := CredentialProjection{
		Kind: kind, KID: credential.KID, KeyID: string(credential.APIKeyID),
		SecretHMAC:    append([]byte(nil), credential.SecretHMAC...),
		PepperVersion: credential.PepperVersion, Status: string(credential.Status),
		NotBefore: credential.NotBefore.UTC(), ExpiresAt: cloneTime(credential.ExpiresAt),
	}
	if delegation != nil {
		projection.ManagementSessionID = delegation.ManagementSessionID
		projection.PrincipalID = delegation.PrincipalID
		projection.DelegationEpoch = delegation.DelegationEpoch
		projection.UserID = delegation.UserID
		projection.TeamID = delegation.TeamID
		projection.Audience = delegation.Audience
	}
	return projection, nil
}

func Compile(candidate Candidate, options CompileOptions) (Projection, error) {
	if candidate.Revision == 0 {
		return Projection{}, fmt.Errorf("projection revision must be positive")
	}
	if err := candidate.Namespace.Validate(); err != nil {
		return Projection{}, fmt.Errorf("namespace: %w", err)
	}
	if candidate.Namespace.Status != accesscontrol.NamespaceStatusActive {
		return Projection{}, fmt.Errorf("namespace is not active")
	}
	if candidate.Key.Status != accesscontrol.APIKeyStatusActive {
		return Projection{}, fmt.Errorf("key is not active")
	}
	if err := accesscontrol.ValidateAPIKeyRelationships(candidate.Key, candidate.Relationships); err != nil {
		return Projection{}, fmt.Errorf("key relationships: %w", err)
	}
	if candidate.Key.NamespaceID != candidate.Namespace.ID {
		return Projection{}, fmt.Errorf("key and namespace do not match")
	}

	access, compileErr := accesscontrol.ResolveAccessBindings(
		candidate.KeyAccessBindings, candidate.UserAccessBindings, candidate.TeamAccessBindings,
	)
	if compileErr != nil {
		return Projection{}, fmt.Errorf("resolve access inheritance: %w", compileErr)
	}
	if err := validateLayerSubjects(candidate); err != nil {
		return Projection{}, err
	}
	grants, compileErr := compileGrants(access, candidate.AccessPolicies)
	if compileErr != nil {
		return Projection{}, compileErr
	}

	rate, compileErr := accesscontrol.ResolveRateBindings(
		candidate.KeyRateBindings, candidate.UserRateBindings, candidate.TeamRateBindings,
	)
	if compileErr != nil {
		return Projection{}, fmt.Errorf("resolve quota inheritance: %w", compileErr)
	}
	rateBindings, compileErr := compileRateBindings(rate, candidate.RatePolicies, options)
	if compileErr != nil {
		return Projection{}, compileErr
	}

	projection := Projection{
		NamespaceID: string(candidate.Namespace.ID), QuotaPartition: string(candidate.Namespace.QuotaPartitionID),
		BillingCurrency: candidate.Namespace.BillingCurrency, KeyID: string(candidate.Key.ID),
		KeyStatus: candidate.Key.Status, KeyExpiresAt: cloneTime(candidate.Key.ExpiresAt),
		Revision: candidate.Revision, PolicyEpoch: candidate.Key.PolicyEpoch,
		DelegationEpoch: candidate.Key.DelegationEpoch, AccessSource: access.Source,
		Grants: grants, RateBindings: rateBindings, RoutingClaims: cloneClaims(candidate.RoutingClaims),
	}
	if candidate.Key.Owner.Kind == accesscontrol.SubjectKindUser {
		projection.UserID = string(candidate.Key.Owner.ID)
		projection.TeamID = string(candidate.Key.ContextTeamID)
	} else {
		projection.TeamID = string(candidate.Key.Owner.ID)
	}
	digest, compileErr := projectionDigest(projection)
	if compileErr != nil {
		return Projection{}, compileErr
	}
	projection.Digest = digest
	return projection, nil
}

func (p Projection) Evaluate(resourceType accesscontrol.GrantResourceType, resourceID string, permission accesscontrol.GrantPermission) accesscontrol.AccessDecision {
	decision := accesscontrol.AccessDecisionDeny
	for _, grant := range p.Grants {
		if grant.ResourceType != resourceType || grant.ResourceID != resourceID || grant.Permission != permission {
			continue
		}
		if grant.Effect == accesscontrol.GrantEffectDeny {
			return accesscontrol.AccessDecisionDeny
		}
		if grant.Effect == accesscontrol.GrantEffectAllow {
			decision = accesscontrol.AccessDecisionAllow
		}
	}
	return decision
}

// VerifyDigest rejects a projection whose serialized contents no longer match
// the digest produced by the control-plane compiler. Runtime readers call this
// after decoding the immutable policy document and before making any decision.
// The expected digest is the value pinned by the active-policy pointer.
func (p Projection) VerifyDigest(expected string) error {
	if expected == "" || p.Digest == "" || p.Digest != expected {
		return fmt.Errorf("access projection digest does not match its active pointer")
	}
	actual, err := projectionDigest(p)
	if err != nil {
		return err
	}
	if actual != expected {
		return fmt.Errorf("access projection contents failed digest verification")
	}
	return nil
}

func compileGrants(effective accesscontrol.EffectiveAccessBindings, policies map[accesscontrol.AccessPolicyID]accesscontrol.AccessPolicy) ([]Grant, error) {
	result := make([]Grant, 0)
	for _, binding := range effective.Bindings {
		policy, ok := policies[binding.PolicyID]
		if !ok || policy.ID != binding.PolicyID || policy.NamespaceID != binding.NamespaceID {
			return nil, fmt.Errorf("access binding %s references an unavailable policy", binding.ID)
		}
		if err := policy.Validate(); err != nil {
			return nil, fmt.Errorf("access policy %s: %w", policy.ID, err)
		}
		if policy.Status != accesscontrol.PolicyStatusActive {
			return nil, fmt.Errorf("access policy %s is not active", policy.ID)
		}
		for _, grant := range policy.Grants {
			result = append(result, Grant{
				BindingID: string(binding.ID), PolicyID: string(policy.ID), Source: effective.Source,
				ResourceType: grant.Resource.Type, ResourceID: string(grant.Resource.ID),
				Permission: grant.Permission, Effect: grant.Effect,
			})
		}
	}
	sort.Slice(result, func(i, j int) bool { return grantKey(result[i]) < grantKey(result[j]) })
	return result, nil
}

func compileRateBindings(
	effective accesscontrol.EffectiveRateBindings,
	policies map[accesscontrol.RateLimitPolicyID]accesscontrol.RateLimitPolicy,
	options CompileOptions,
) ([]RateBinding, error) {
	resolved := make([]accesscontrol.ResolvedRateBinding, 0, len(effective.HardCaps)+1)
	if effective.Allocation != nil {
		resolved = append(resolved, *effective.Allocation)
	}
	resolved = append(resolved, effective.HardCaps...)
	result := make([]RateBinding, 0, len(resolved))
	for _, item := range resolved {
		binding := item.Binding
		policy, ok := policies[binding.PolicyID]
		if !ok || policy.ID != binding.PolicyID || policy.NamespaceID != binding.NamespaceID {
			return nil, fmt.Errorf("rate binding %s references an unavailable policy", binding.ID)
		}
		if err := policy.Validate(); err != nil {
			return nil, fmt.Errorf("rate policy %s: %w", policy.ID, err)
		}
		if policy.Status != accesscontrol.PolicyStatusActive {
			return nil, fmt.Errorf("rate policy %s is not active", policy.ID)
		}
		rules := make([]ProjectedRateRule, 0, len(policy.Rules))
		for _, rule := range policy.Rules {
			compiled, err := compileRule(rule)
			if err != nil {
				return nil, fmt.Errorf("rate policy %s rule %s: %w", policy.ID, rule.ID, err)
			}
			projected := ProjectedRateRule{Rule: compiled}
			if compiled.Algorithm == quota.AlgorithmCalendarWindow {
				projected.TZDBVersion, projected.CalendarSchedule, err = compileCalendarSchedule(compiled, options)
				if err != nil {
					return nil, fmt.Errorf("rate policy %s rule %s calendar schedule: %w", policy.ID, rule.ID, err)
				}
			}
			rules = append(rules, projected)
		}
		sort.Slice(rules, func(i, j int) bool {
			if rules[i].Rule.Ordinal == rules[j].Rule.Ordinal {
				return rules[i].Rule.ID < rules[j].Rule.ID
			}
			return rules[i].Rule.Ordinal < rules[j].Rule.Ordinal
		})
		result = append(result, RateBinding{
			BindingID: string(binding.ID), PolicyID: string(binding.PolicyID),
			SubjectID: string(binding.Subject.ID), Source: item.Source, Mode: binding.Mode, Rules: rules,
		})
	}
	sort.Slice(result, func(i, j int) bool { return result[i].BindingID < result[j].BindingID })
	return result, nil
}

func compileRule(rule accesscontrol.RateLimitRule) (quota.RateLimitRule, error) {
	if err := rule.Validate(); err != nil {
		return quota.RateLimitRule{}, err
	}
	compiled := quota.RateLimitRule{
		ID: string(rule.ID), Metric: quota.Metric(rule.Metric), Algorithm: quota.Algorithm(rule.Algorithm),
		Accounting: quota.Accounting(rule.Accounting), Enforcement: quota.Enforcement(rule.Enforcement),
		Ordinal: int(rule.Ordinal), Window: rule.Window, CalendarPeriod: quota.CalendarPeriod(rule.CalendarPeriod),
		CalendarTimezone: rule.Timezone, RefillPeriod: rule.RefillPeriod,
		GCRAEmissionInterval: rule.GCRAEmissionInterval,
	}
	var err error
	if rule.Limit != "" {
		if rule.Metric == accesscontrol.RateMetricCost {
			value, parseErr := quota.ParseCurrencyDecimal(string(rule.Limit))
			if parseErr != nil {
				return quota.RateLimitRule{}, parseErr
			}
			compiled.CostLimit = &value
		} else {
			value, parseErr := quota.ParseQuotaInteger(string(rule.Limit))
			if parseErr != nil {
				return quota.RateLimitRule{}, parseErr
			}
			compiled.WholeLimit = &value
		}
	}
	compiled.BucketCapacity, err = parseOptionalInteger(rule.BucketCapacity)
	if err != nil {
		return quota.RateLimitRule{}, err
	}
	compiled.RefillAmount, err = parseOptionalInteger(rule.RefillAmount)
	if err != nil {
		return quota.RateLimitRule{}, err
	}
	if rule.GCRABurstTolerance != nil {
		value, parseErr := quota.ParseQuotaInteger(fmt.Sprintf("%d", *rule.GCRABurstTolerance))
		if parseErr != nil {
			return quota.RateLimitRule{}, parseErr
		}
		compiled.GCRABurstTolerance = &value
	}
	if err := compiled.Validate(); err != nil {
		return quota.RateLimitRule{}, err
	}
	return compiled, nil
}

const calendarHorizonMonths = 18

func compileCalendarSchedule(rule quota.RateLimitRule, options CompileOptions) (string, []CalendarInterval, error) {
	start := options.CalendarScheduleStart
	if start.IsZero() || start.Location() != time.UTC || start.Nanosecond()%int(time.Millisecond) != 0 {
		return "", nil, fmt.Errorf("calendar schedule start must be millisecond-aligned UTC")
	}
	location, err := time.LoadLocation(rule.CalendarTimezone)
	if err != nil {
		return "", nil, fmt.Errorf("load timezone %q: %w", rule.CalendarTimezone, err)
	}
	local := start.In(location)
	var boundary time.Time
	switch rule.CalendarPeriod {
	case quota.CalendarPeriodDay:
		boundary = time.Date(local.Year(), local.Month(), local.Day(), 0, 0, 0, 0, location)
	case quota.CalendarPeriodMonth:
		boundary = time.Date(local.Year(), local.Month(), 1, 0, 0, 0, 0, location)
	default:
		return "", nil, fmt.Errorf("unsupported calendar period %q", rule.CalendarPeriod)
	}
	horizon := start.AddDate(0, calendarHorizonMonths, 0)
	intervals := make([]CalendarInterval, 0, 560)
	for boundary.UTC().Before(horizon) {
		var next time.Time
		if rule.CalendarPeriod == quota.CalendarPeriodDay {
			next = boundary.AddDate(0, 0, 1)
		} else {
			next = boundary.AddDate(0, 1, 0)
		}
		from, to := boundary.UTC(), next.UTC()
		if !from.Before(to) {
			return "", nil, fmt.Errorf("timezone produced an empty or reversed interval")
		}
		intervals = append(intervals, CalendarInterval{Start: from, End: to})
		boundary = next
	}
	if len(intervals) == 0 || intervals[0].Start.After(start) || intervals[len(intervals)-1].End.Before(horizon) {
		return "", nil, fmt.Errorf("compiled schedule does not cover the required horizon")
	}
	return EmbeddedTZDBVersion, intervals, nil
}

func parseOptionalInteger(value accesscontrol.QuotaValue) (*quota.QuotaInteger, error) {
	if value == "" {
		return nil, nil
	}
	parsed, err := quota.ParseQuotaInteger(string(value))
	if err != nil {
		return nil, err
	}
	return &parsed, nil
}

func validateLayerSubjects(candidate Candidate) error {
	keySubject := candidate.Key.SubjectRef()
	if err := requireSubject(candidate.KeyAccessBindings, keySubject); err != nil {
		return fmt.Errorf("key access bindings: %w", err)
	}
	if err := requireRateSubject(candidate.KeyRateBindings, keySubject); err != nil {
		return fmt.Errorf("key rate bindings: %w", err)
	}
	var userSubject *accesscontrol.SubjectRef
	var teamSubject *accesscontrol.SubjectRef
	if candidate.Key.Owner.Kind == accesscontrol.SubjectKindUser {
		value := candidate.Relationships.OwnerUser.SubjectRef()
		userSubject = &value
		if candidate.Relationships.ContextTeam != nil {
			value := candidate.Relationships.ContextTeam.SubjectRef()
			teamSubject = &value
		}
	} else {
		value := candidate.Relationships.OwnerTeam.SubjectRef()
		teamSubject = &value
	}
	if err := requireOptionalSubject(candidate.UserAccessBindings, userSubject); err != nil {
		return fmt.Errorf("user access bindings: %w", err)
	}
	if err := requireOptionalRateSubject(candidate.UserRateBindings, userSubject); err != nil {
		return fmt.Errorf("user rate bindings: %w", err)
	}
	if err := requireOptionalSubject(candidate.TeamAccessBindings, teamSubject); err != nil {
		return fmt.Errorf("team access bindings: %w", err)
	}
	if err := requireOptionalRateSubject(candidate.TeamRateBindings, teamSubject); err != nil {
		return fmt.Errorf("team rate bindings: %w", err)
	}
	return nil
}

func requireSubject(bindings []accesscontrol.AccessPolicyBinding, subject accesscontrol.SubjectRef) error {
	for _, binding := range bindings {
		if binding.Subject != subject {
			return fmt.Errorf("binding %s targets a different subject", binding.ID)
		}
	}
	return nil
}

func requireOptionalSubject(bindings []accesscontrol.AccessPolicyBinding, subject *accesscontrol.SubjectRef) error {
	if subject == nil && len(bindings) > 0 {
		return fmt.Errorf("bindings exist without an applicable subject")
	}
	if subject == nil {
		return nil
	}
	return requireSubject(bindings, *subject)
}

func requireRateSubject(bindings []accesscontrol.RateLimitBinding, subject accesscontrol.SubjectRef) error {
	for _, binding := range bindings {
		if binding.Subject != subject {
			return fmt.Errorf("binding %s targets a different subject", binding.ID)
		}
	}
	return nil
}

func requireOptionalRateSubject(bindings []accesscontrol.RateLimitBinding, subject *accesscontrol.SubjectRef) error {
	if subject == nil && len(bindings) > 0 {
		return fmt.Errorf("bindings exist without an applicable subject")
	}
	if subject == nil {
		return nil
	}
	return requireRateSubject(bindings, *subject)
}

func projectionDigest(projection Projection) (string, error) {
	projection.Digest = ""
	payload, err := json.Marshal(projection)
	if err != nil {
		return "", fmt.Errorf("marshal access projection: %w", err)
	}
	digest := sha256.Sum256(payload)
	return hex.EncodeToString(digest[:]), nil
}

func grantKey(grant Grant) string {
	return grant.BindingID + "\x00" + grant.PolicyID + "\x00" + string(grant.ResourceType) + "\x00" +
		grant.ResourceID + "\x00" + string(grant.Permission) + "\x00" + string(grant.Effect)
}

func cloneClaims(input map[string]routingsnapshot.ClaimValue) map[string]routingsnapshot.ClaimValue {
	if len(input) == 0 {
		return nil
	}
	result := make(map[string]routingsnapshot.ClaimValue, len(input))
	for name, value := range input {
		result[name] = value
	}
	return result
}

func cloneTime(value *time.Time) *time.Time {
	if value == nil {
		return nil
	}
	copy := value.UTC()
	return &copy
}
