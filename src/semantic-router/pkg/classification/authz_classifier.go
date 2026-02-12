package classification

import (
	"fmt"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
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
	kind string // "user" or "group" (already lowercased)
	name string // already trimmed
}

// AuthzClassifier evaluates user identity and group membership against RBAC role bindings.
// It follows the Kubernetes RoleBinding pattern:
//   - Subject  → user ID + groups (from auth backend: Authorino, Envoy Gateway JWT, etc.)
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
//   - Each subject must have kind "User" or "Group" (case-insensitive)
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
				"add at least one subject with kind User or Group", rb.Name)
		}

		nb := normalizedBinding{
			name:     rb.Name,
			role:     rb.Role,
			subjects: make([]normalizedSubject, 0, len(rb.Subjects)),
		}

		for i, s := range rb.Subjects {
			kind := strings.ToLower(strings.TrimSpace(s.Kind))
			if kind != "user" && kind != "group" {
				return nil, fmt.Errorf("role_bindings: binding %q subject[%d] has invalid kind %q — "+
					"must be \"User\" or \"Group\"", rb.Name, i, s.Kind)
			}
			name := strings.TrimSpace(s.Name)
			if name == "" {
				return nil, fmt.Errorf("role_bindings: binding %q subject[%d] (kind: %s) has empty name — "+
					"the name must match the value your auth backend injects in the identity headers (configured via authz.identity)",
					rb.Name, i, s.Kind)
			}
			nb.subjects = append(nb.subjects, normalizedSubject{kind: kind, name: name})
		}

		normalized = append(normalized, nb)
	}

	return &AuthzClassifier{bindings: normalized}, nil
}

// Classify evaluates the RBAC role bindings against the user identity and groups.
//
// Match logic: a binding matches if ANY of its subjects match:
//   - kind: "user"  → matches if subject.name == userID
//   - kind: "group" → matches if subject.name is in userGroups
//
// When a binding matches, its role is emitted as the signal name.
// Multiple bindings can match. If multiple bindings grant the same role, it is deduplicated.
//
// Returns an error if userID is empty and role bindings are configured — this prevents
// silent bypass when ext_authz fails to inject the user identity header.
func (c *AuthzClassifier) Classify(userID string, userGroups []string) (*AuthzResult, error) {
	if len(c.bindings) == 0 {
		return &AuthzResult{}, nil
	}

	if userID == "" {
		return nil, fmt.Errorf("authz signal: user identity header is empty but %d role_bindings are configured — "+
			"ensure your auth backend injects the user identity header (check authz.identity.user_id_header config); "+
			"refusing to evaluate without user identity (no silent bypass allowed)", len(c.bindings))
	}

	// Deduplicate roles (multiple bindings can grant the same role)
	roleSet := make(map[string]bool)
	var matchedRoles []string

	for _, rb := range c.bindings {
		matched := false

		for _, s := range rb.subjects {
			// Kind and Name are already normalized at startup — no runtime normalization needed
			switch s.kind {
			case "user":
				if s.name == userID {
					matched = true
					logging.Infof("[Authz Signal] Binding %q matched: subject User %q == user ID %q → role %q",
						rb.name, s.name, userID, rb.role)
				}
			case "group":
				for _, ug := range userGroups {
					if s.name == ug {
						matched = true
						logging.Infof("[Authz Signal] Binding %q matched: subject Group %q in user groups → role %q",
							rb.name, s.name, rb.role)
						break
					}
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
		logging.Infof("[Authz Signal] No roles matched for user %q (groups: %v)", userID, userGroups)
	} else {
		logging.Infof("[Authz Signal] Matched %d roles for user %q: %v", len(matchedRoles), userID, matchedRoles)
	}

	return &AuthzResult{
		MatchedRules: matchedRoles,
	}, nil
}

// ParseUserGroups parses a comma-separated groups header value into a slice of group names.
// Whitespace around group names is trimmed. Empty strings are excluded.
func ParseUserGroups(headerValue string) []string {
	if headerValue == "" {
		return nil
	}
	parts := strings.Split(headerValue, ",")
	var groups []string
	for _, p := range parts {
		g := strings.TrimSpace(p)
		if g != "" {
			groups = append(groups, g)
		}
	}
	return groups
}
