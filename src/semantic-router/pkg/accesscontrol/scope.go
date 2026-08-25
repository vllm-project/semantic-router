package accesscontrol

type ScopeKind string

const (
	ScopeKindCluster   ScopeKind = "cluster"
	ScopeKindNamespace ScopeKind = "namespace"
	ScopeKindTeam      ScopeKind = "team"
	ScopeKindUser      ScopeKind = "user"
	ScopeKindResource  ScopeKind = "resource"
)

func (k ScopeKind) Valid() bool {
	switch k {
	case ScopeKindCluster, ScopeKindNamespace, ScopeKindTeam, ScopeKindUser, ScopeKindResource:
		return true
	default:
		return false
	}
}

type ScopeResourceType string

const (
	ScopeResourceAPIKey ScopeResourceType = "api_key"
	// #nosec G101 -- this is a resource-type identifier, not a credential value.
	ScopeResourceAPIKeyCredential          ScopeResourceType = "api_key_credential"
	ScopeResourceDelegatedInferenceSession ScopeResourceType = "delegated_inference_session"
	ScopeResourceMembership                ScopeResourceType = "membership"
	ScopeResourcePrincipalUserLink         ScopeResourceType = "principal_user_link"
	ScopeResourceAccessPolicy              ScopeResourceType = "access_policy"
	ScopeResourceAccessPolicyGrant         ScopeResourceType = "access_policy_grant"
	ScopeResourceAccessPolicyBinding       ScopeResourceType = "access_policy_binding"
	ScopeResourceRateLimitPolicy           ScopeResourceType = "rate_limit_policy"
	ScopeResourceRateLimitRule             ScopeResourceType = "rate_limit_rule"
	ScopeResourceRateLimitBinding          ScopeResourceType = "rate_limit_binding"
	ScopeResourceModel                     ScopeResourceType = "model"
	ScopeResourceRecipe                    ScopeResourceType = "recipe"
	ScopeResourceEntrypoint                ScopeResourceType = "entrypoint"
	ScopeResourceProviderCredential        ScopeResourceType = "provider_credential"
	ScopeResourceManagementRole            ScopeResourceType = "management_role"
	ScopeResourceServiceAccount            ScopeResourceType = "service_account"
	ScopeResourceInvitation                ScopeResourceType = "invitation"
	ScopeResourceOperation                 ScopeResourceType = "operation"
	ScopeResourceUnknownUsageFence         ScopeResourceType = "unknown_usage_fence"
	ScopeResourceUsage                     ScopeResourceType = "usage"
	ScopeResourceLog                       ScopeResourceType = "log"
	ScopeResourceAgentProfile              ScopeResourceType = "agent_profile"
	ScopeResourceAgentSkill                ScopeResourceType = "agent_skill"
	// #nosec G101 -- this is a resource-type identifier, not a credential value.
	ScopeResourceAgentToolCredential ScopeResourceType = "agent_tool_credential"
	ScopeResourceAgentToolSource     ScopeResourceType = "agent_tool_source"
	ScopeResourceAgentSession        ScopeResourceType = "agent_session"
	ScopeResourceAgentTurn           ScopeResourceType = "agent_turn"
	ScopeResourceAgentArtifact       ScopeResourceType = "agent_artifact"
	ScopeResourcePublicationPlan     ScopeResourceType = "publication_plan"
)

func (t ScopeResourceType) Valid() bool {
	switch t {
	case ScopeResourceAPIKey, ScopeResourceAPIKeyCredential,
		ScopeResourceDelegatedInferenceSession,
		ScopeResourceMembership, ScopeResourcePrincipalUserLink,
		ScopeResourceAccessPolicy, ScopeResourceAccessPolicyGrant,
		ScopeResourceAccessPolicyBinding,
		ScopeResourceRateLimitPolicy, ScopeResourceRateLimitRule,
		ScopeResourceRateLimitBinding,
		ScopeResourceModel, ScopeResourceRecipe, ScopeResourceEntrypoint,
		ScopeResourceProviderCredential, ScopeResourceManagementRole,
		ScopeResourceServiceAccount, ScopeResourceInvitation,
		ScopeResourceOperation, ScopeResourceUnknownUsageFence,
		ScopeResourceUsage, ScopeResourceLog,
		ScopeResourceAgentProfile, ScopeResourceAgentSkill, ScopeResourceAgentToolCredential,
		ScopeResourceAgentToolSource, ScopeResourceAgentSession,
		ScopeResourceAgentTurn, ScopeResourceAgentArtifact, ScopeResourcePublicationPlan:
		return true
	default:
		return false
	}
}

// Scope is a discriminated value. Fields not selected by Kind must be empty.
type Scope struct {
	Kind         ScopeKind
	NamespaceID  NamespaceID
	TeamID       TeamID
	UserID       UserID
	ResourceType ScopeResourceType
	ResourceID   ResourceID
}

func ClusterScope() Scope { return Scope{Kind: ScopeKindCluster} }

func NamespaceScope(namespaceID NamespaceID) Scope {
	return Scope{Kind: ScopeKindNamespace, NamespaceID: namespaceID}
}

func TeamScope(namespaceID NamespaceID, teamID TeamID) Scope {
	return Scope{Kind: ScopeKindTeam, NamespaceID: namespaceID, TeamID: teamID}
}

func UserScope(namespaceID NamespaceID, userID UserID) Scope {
	return Scope{Kind: ScopeKindUser, NamespaceID: namespaceID, UserID: userID}
}

func ResourceScope(namespaceID NamespaceID, resourceType ScopeResourceType, resourceID ResourceID) Scope {
	return Scope{
		Kind: ScopeKindResource, NamespaceID: namespaceID,
		ResourceType: resourceType, ResourceID: resourceID,
	}
}

func (s Scope) Validate() error {
	if !s.Kind.Valid() {
		return invalid("kind", "is not a valid scope kind")
	}
	switch s.Kind {
	case ScopeKindCluster:
		return s.validateCluster()
	case ScopeKindNamespace:
		return s.validateNamespace()
	case ScopeKindTeam:
		return s.validateTeam()
	case ScopeKindUser:
		return s.validateUser()
	case ScopeKindResource:
		return s.validateResource()
	}
	return nil
}

func (s Scope) validateCluster() error {
	if s.NamespaceID != "" || s.TeamID != "" || s.UserID != "" || s.ResourceType != "" || s.ResourceID != "" {
		return invalid("scope", "cluster scope cannot carry namespace or resource fields")
	}
	return nil
}

func (s Scope) validateNamespace() error {
	return joinValidation(
		validateRequired("namespace_id", string(s.NamespaceID)),
		validateScopeEmpty(s.TeamID == "" && s.UserID == "" && s.ResourceType == "" && s.ResourceID == ""),
	)
}

func (s Scope) validateTeam() error {
	return joinValidation(
		validateRequired("namespace_id", string(s.NamespaceID)),
		validateRequired("team_id", string(s.TeamID)),
		validateScopeEmpty(s.UserID == "" && s.ResourceType == "" && s.ResourceID == ""),
	)
}

func (s Scope) validateUser() error {
	return joinValidation(
		validateRequired("namespace_id", string(s.NamespaceID)),
		validateRequired("user_id", string(s.UserID)),
		validateScopeEmpty(s.TeamID == "" && s.ResourceType == "" && s.ResourceID == ""),
	)
}

func (s Scope) validateResource() error {
	var typeErr error
	if !s.ResourceType.Valid() {
		typeErr = invalid("resource_type", "is not a registered scope resource type")
	}
	return joinValidation(
		validateRequired("namespace_id", string(s.NamespaceID)),
		typeErr,
		validateRequired("resource_id", string(s.ResourceID)),
		validateScopeEmpty(s.TeamID == "" && s.UserID == ""),
	)
}

func validateScopeEmpty(valid bool) error {
	if !valid {
		return invalid("scope", "contains fields from another scope variant")
	}
	return nil
}

// ScopedTarget is assembled from the authoritative resource graph. Ancestors
// encode only schema-declared ownership, such as a Team-owned key or an API-key
// credential. Membership alone must never be supplied as a Team ancestor for a
// User-owned resource.
type ScopedTarget struct {
	Scope     Scope
	Ancestors []Scope
}

func (t ScopedTarget) Validate() error {
	if err := t.Scope.Validate(); err != nil {
		return err
	}
	if t.Scope.Kind != ScopeKindResource && len(t.Ancestors) > 0 {
		return invalid("ancestors", "are valid only for a resource target")
	}
	seen := make(map[Scope]struct{}, len(t.Ancestors))
	for _, ancestor := range t.Ancestors {
		if err := ancestor.Validate(); err != nil {
			return invalid("ancestors", err.Error())
		}
		if ancestor.Kind == ScopeKindCluster || ancestor.Kind == ScopeKindNamespace {
			return invalid("ancestors", "cluster and namespace containment is implicit")
		}
		if ancestor.NamespaceID != t.Scope.NamespaceID {
			return invalid("ancestors", "must stay within the target namespace")
		}
		if ancestor == t.Scope {
			return invalid("ancestors", "must not repeat the target scope")
		}
		if _, exists := seen[ancestor]; exists {
			return invalid("ancestors", "must not contain duplicates")
		}
		seen[ancestor] = struct{}{}
	}
	return nil
}

// Contains reports scope containment. Namespace containment is structural;
// narrower resource ancestry is accepted only from a validated ScopedTarget.
func (s Scope) Contains(target ScopedTarget) bool {
	if s.Validate() != nil || target.Validate() != nil {
		return false
	}
	if s.Kind == ScopeKindCluster {
		return true
	}
	if target.Scope.Kind == ScopeKindCluster {
		return false
	}
	if s.Kind == ScopeKindNamespace {
		return s.NamespaceID == target.Scope.NamespaceID
	}
	if s == target.Scope {
		return true
	}
	for _, ancestor := range target.Ancestors {
		if s == ancestor {
			return true
		}
	}
	return false
}

func (s Scope) ContainsAll(targets []ScopedTarget) bool {
	for _, target := range targets {
		if !s.Contains(target) {
			return false
		}
	}
	return true
}
