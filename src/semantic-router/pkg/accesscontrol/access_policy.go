package accesscontrol

import (
	"strconv"
	"time"
)

type GrantResourceType string

const (
	GrantResourceEntrypoint GrantResourceType = "entrypoint"
	GrantResourceModel      GrantResourceType = "model"
)

func (t GrantResourceType) Valid() bool {
	return t == GrantResourceEntrypoint || t == GrantResourceModel
}

type GrantPermission string

const (
	GrantPermissionDiscover GrantPermission = "discover"
	GrantPermissionInvoke   GrantPermission = "invoke"
)

func (p GrantPermission) Valid() bool {
	return p == GrantPermissionDiscover || p == GrantPermissionInvoke
}

type GrantEffect string

const (
	GrantEffectAllow GrantEffect = "allow"
	GrantEffectDeny  GrantEffect = "deny"
)

func (e GrantEffect) Valid() bool { return e == GrantEffectAllow || e == GrantEffectDeny }

type GrantResource struct {
	Type GrantResourceType
	ID   ResourceID
}

func (r GrantResource) Validate() error {
	var typeErr error
	if !r.Type.Valid() {
		typeErr = invalid("resource_type", "is not a valid grant resource type")
	}
	return joinValidation(typeErr, validateExplicitResourceID(r.ID))
}

func validateExplicitResourceID(id ResourceID) error {
	if err := validateRequired("resource_id", string(id)); err != nil {
		return err
	}
	for _, char := range string(id) {
		switch char {
		case '*', '?', '[', ']':
			return invalid("resource_id", "must be an explicit immutable resource UID, not a pattern")
		}
	}
	return nil
}

type AccessPolicyGrant struct {
	PolicyID   AccessPolicyID
	Resource   GrantResource
	Permission GrantPermission
	Effect     GrantEffect
}

func (g AccessPolicyGrant) Validate() error {
	var permissionErr, effectErr error
	if !g.Permission.Valid() {
		permissionErr = invalid("permission", "is not discover or invoke")
	}
	if !g.Effect.Valid() {
		effectErr = invalid("effect", "is not allow or deny")
	}
	return joinValidation(
		validateRequired("policy_id", string(g.PolicyID)),
		g.Resource.Validate(),
		permissionErr,
		effectErr,
	)
}

type AccessPolicy struct {
	NamespaceID NamespaceID
	ID          AccessPolicyID
	DisplayName string
	Status      PolicyStatus
	Revision    Revision
	Grants      []AccessPolicyGrant
	CreatedAt   time.Time
	UpdatedAt   time.Time
}

func (p AccessPolicy) Validate() error {
	var statusErr error
	if !p.Status.Valid() {
		statusErr = invalid("status", "is not a valid policy status")
	}
	errs := []error{
		validateRequired("namespace_id", string(p.NamespaceID)),
		validateRequired("id", string(p.ID)),
		validateRequired("display_name", p.DisplayName),
		statusErr,
		validateRevision(p.Revision),
		validateTimestamps(p.CreatedAt, p.UpdatedAt),
	}
	seen := make(map[accessGrantKey]struct{}, len(p.Grants))
	for index, grant := range p.Grants {
		if grant.PolicyID != p.ID {
			errs = append(errs, invalid("grants", "grant policy_id must match its parent policy"))
		}
		if err := grant.Validate(); err != nil {
			errs = append(errs, invalid("grants", "grant at index "+strconv.Itoa(index)+": "+err.Error()))
		}
		key := grant.key()
		if _, exists := seen[key]; exists {
			errs = append(errs, invalid("grants", "contains a duplicate grant tuple"))
		}
		seen[key] = struct{}{}
	}
	return joinValidation(errs...)
}

type accessGrantKey struct {
	ResourceType GrantResourceType
	ResourceID   ResourceID
	Permission   GrantPermission
	Effect       GrantEffect
}

func (g AccessPolicyGrant) key() accessGrantKey {
	return accessGrantKey{
		ResourceType: g.Resource.Type,
		ResourceID:   g.Resource.ID,
		Permission:   g.Permission,
		Effect:       g.Effect,
	}
}

type AccessDecision string

const (
	AccessDecisionAllow AccessDecision = "allow"
	AccessDecisionDeny  AccessDecision = "deny"
)

type GrantEvaluation struct {
	Decision AccessDecision
	Matched  []AccessPolicyGrant
}

// EvaluateGrants applies exact-UID matching, allow union, and deny precedence.
// An empty or unmatched set denies access.
func EvaluateGrants(grants []AccessPolicyGrant, resource GrantResource, permission GrantPermission) GrantEvaluation {
	result := GrantEvaluation{Decision: AccessDecisionDeny}
	hasAllow := false
	for _, grant := range grants {
		if grant.Resource != resource || grant.Permission != permission {
			continue
		}
		result.Matched = append(result.Matched, grant)
		if grant.Effect == GrantEffectDeny {
			return result
		}
		if grant.Effect == GrantEffectAllow {
			hasAllow = true
		}
	}
	if hasAllow {
		result.Decision = AccessDecisionAllow
	}
	return result
}
