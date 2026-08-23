package classification

import (
	"fmt"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routingsnapshot"
)

// AuthzResult represents the result of authz signal classification.
// It contains only the matched role names — the decision engine uses these
// to select models via modelRefs.
type AuthzResult struct {
	// MatchedRules contains the role names from all matched role bindings.
	// These are the Role field values, not the binding Name field.
	MatchedRules []string
}

// normalizedBinding is the internal representation with Kind/Name already normalized.
// This avoids repeated string normalization at request time.
type normalizedBinding struct {
	name     string // binding name (for logs)
	role     string // role name (emitted as signal)
	subjects []normalizedSubject
}

type normalizedSubject struct {
	kind string // "user", "team", or "group" (already lowercased)
	name string // already trimmed
}

// AuthzClassifier evaluates authenticated TenantContext facts against routing
// role bindings. These roles affect routing only and grant no access permission.
//   - Subject  → user ID, team ID, or a compiled routing-claim membership
//   - Role     → RoleBinding.Role (emitted as the signal name)
//   - Permission → decision engine modelRefs (not this classifier's concern)
type AuthzClassifier struct {
	bindings []normalizedBinding
}

// NewAuthzClassifier creates a new AuthzClassifier from RBAC role bindings.
// All validation and normalization happens here at startup. If this function returns
// without error, Classify() is guaranteed to work correctly at request time.
//
// Validates at startup:
//   - Binding name must not be empty
//   - Binding name must be unique across all bindings
//   - Role must not be empty
//   - At least one subject must be specified
//   - Each subject must have kind "User", "Team", or "Group" (case-insensitive)
//   - Each subject must have a non-empty name (whitespace-only is rejected)
//
// Normalizes at startup:
//   - Subject.Kind is lowercased and trimmed
//   - Subject.Name is trimmed (preserving original case for exact matching)
func NewAuthzClassifier(bindings []config.RoleBinding) (*AuthzClassifier, error) {
	seenNames := make(map[string]bool, len(bindings))
	normalized := make([]normalizedBinding, 0, len(bindings))

	for _, rb := range bindings {
		if rb.Name == "" {
			return nil, fmt.Errorf("role_bindings: binding with empty name is not allowed")
		}
		if seenNames[rb.Name] {
			return nil, fmt.Errorf("role_bindings: duplicate binding name %q — "+
				"each binding must have a unique name for audit log clarity", rb.Name)
		}
		seenNames[rb.Name] = true

		if rb.Role == "" {
			return nil, fmt.Errorf("role_bindings: binding %q has empty role — "+
				"set the role field to the name used in decision conditions (type: \"authz\", name: \"<role>\")", rb.Name)
		}
		if len(rb.Subjects) == 0 {
			return nil, fmt.Errorf("role_bindings: binding %q has no subjects — "+
				"add at least one subject with kind User, Team, or Group", rb.Name)
		}

		nb := normalizedBinding{
			name:     rb.Name,
			role:     rb.Role,
			subjects: make([]normalizedSubject, 0, len(rb.Subjects)),
		}

		for i, s := range rb.Subjects {
			kind := strings.ToLower(strings.TrimSpace(s.Kind))
			if kind != "user" && kind != "team" && kind != "group" {
				return nil, fmt.Errorf("role_bindings: binding %q subject[%d] has invalid kind %q — "+
					"must be \"User\", \"Team\", or \"Group\"", rb.Name, i, s.Kind)
			}
			name := strings.TrimSpace(s.Name)
			if name == "" {
				return nil, fmt.Errorf("role_bindings: binding %q subject[%d] (kind: %s) has empty name",
					rb.Name, i, s.Kind)
			}
			nb.subjects = append(nb.subjects, normalizedSubject{kind: kind, name: name})
		}

		normalized = append(normalized, nb)
	}

	return &AuthzClassifier{bindings: normalized}, nil
}

// Classify evaluates role bindings against authenticated routing identity.
//
// Match logic: a binding matches if ANY of its subjects match:
//   - kind: "user"  → matches the TenantContext user
//   - kind: "team"  → matches the TenantContext team
//   - kind: "group" → matches a string claim value or a true boolean claim name
//
// When a binding matches, its role is emitted as the signal name.
// Multiple bindings can match. If multiple bindings grant the same role, it is deduplicated.
func (c *AuthzClassifier) Classify(identity TrustedRoutingIdentity) (*AuthzResult, error) {
	if len(c.bindings) == 0 {
		return &AuthzResult{}, nil
	}

	if identity.UserID == "" && identity.TeamID == "" && len(identity.Claims) == 0 {
		return nil, fmt.Errorf("authz signal requires an authenticated TenantContext when %d role_bindings are configured", len(c.bindings))
	}
	groups := trustedClaimMemberships(identity.Claims)

	// Deduplicate roles (multiple bindings can grant the same role)
	roleSet := make(map[string]bool)
	var matchedRoles []string

	for _, rb := range c.bindings {
		matched := false

		for _, s := range rb.subjects {
			// Kind and Name are already normalized at startup — no runtime normalization needed
			switch s.kind {
			case "user":
				if s.name == identity.UserID {
					matched = true
					logging.Infof("[Authz Signal] Binding %q matched authenticated User → role %q", rb.name, rb.role)
				}
			case "team":
				if s.name == identity.TeamID {
					matched = true
					logging.Infof("[Authz Signal] Binding %q matched authenticated Team → role %q", rb.name, rb.role)
				}
			case "group":
				if _, ok := groups[s.name]; ok {
					matched = true
					logging.Infof("[Authz Signal] Binding %q matched trusted claim membership → role %q", rb.name, rb.role)
				}
			default:
				// Cannot happen: NewAuthzClassifier validates kind at startup.
				// If it does happen, it's a programming error — fail loudly.
				panic(fmt.Sprintf("authz classifier: unexpected subject kind %q in binding %q — "+
					"this is a bug: NewAuthzClassifier should have rejected this at startup", s.kind, rb.name))
			}
			if matched {
				break
			}
		}

		if matched && !roleSet[rb.role] {
			roleSet[rb.role] = true
			matchedRoles = append(matchedRoles, rb.role)
		}
	}

	if len(matchedRoles) == 0 {
		logging.Infof("[Authz Signal] No routing roles matched authenticated identity")
	} else {
		logging.Infof("[Authz Signal] Matched %d routing roles: %v", len(matchedRoles), matchedRoles)
	}

	return &AuthzResult{
		MatchedRules: matchedRoles,
	}, nil
}

func trustedClaimMemberships(claims map[string]routingsnapshot.ClaimValue) map[string]struct{} {
	result := make(map[string]struct{})
	for name, value := range claims {
		switch value.Kind {
		case "string":
			if member := strings.TrimSpace(value.String); member != "" {
				result[member] = struct{}{}
			}
		case "boolean":
			if value.Boolean {
				result[name] = struct{}{}
			}
		}
	}
	return result
}
