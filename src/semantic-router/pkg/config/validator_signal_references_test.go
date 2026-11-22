package config

import (
	"testing"

	"github.com/stretchr/testify/require"
)

// TestValidateDecisionSignalReferences_WhitespaceRejected verifies that
// strict recipe validation rejects decision signal references containing
// surrounding whitespace, matching the runtime decision engine's exact-match
// behavior. See issue #4984.
func TestValidateDecisionSignalReferences_WhitespaceRejected(t *testing.T) {
	tests := []struct {
		name      string
		node      RuleNode
		wantError string
	}{
		{
			name:      "leading whitespace on keyword name",
			node:      RuleNode{Type: SignalTypeKeyword, Name: " billing"},
			wantError: `recipes[test-recipe].decisions["test-decision"]: signal keyword reference " billing" has surrounding whitespace; expected "billing" without padding`,
		},
		{
			name:      "trailing whitespace on keyword name",
			node:      RuleNode{Type: SignalTypeKeyword, Name: "billing "},
			wantError: `recipes[test-recipe].decisions["test-decision"]: signal keyword reference "billing " has surrounding whitespace; expected "billing" without padding`,
		},
		{
			name:      "both sides whitespace on keyword name",
			node:      RuleNode{Type: SignalTypeKeyword, Name: " billing "},
			wantError: `recipes[test-recipe].decisions["test-decision"]: signal keyword reference " billing " has surrounding whitespace; expected "billing" without padding`,
		},
		{
			name:      "tab character on domain name",
			node:      RuleNode{Type: SignalTypeDomain, Name: "\tucla"},
			wantError: `recipes[test-recipe].decisions["test-decision"]: signal domain reference "\tucla" has surrounding whitespace; expected "ucla" without padding`,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			cfg := &RouterConfig{
				RoutingScope: "test-recipe", // triggers strict references
				IntelligentRouting: IntelligentRouting{
					Signals: Signals{
						KeywordRules: []KeywordRule{{Name: "billing"}},
						Categories:   []Category{testDomainCategory("ucla")},
					},
					Decisions: []Decision{{
						Name:  "test-decision",
						Rules: RuleCombination{Conditions: []RuleNode{tc.node}},
					}},
				},
			}

			err := validateDecisionSignalReferences(cfg)
			require.Error(t, err)
			require.Equal(t, tc.wantError, err.Error())
		})
	}
}

// TestValidateDecisionSignalReferences_ValidNoWhitespace verifies that
// properly-formatted signal references (no surrounding whitespace) still
// pass strict validation.
func TestValidateDecisionSignalReferences_ValidNoWhitespace(t *testing.T) {
	cfg := &RouterConfig{
		RoutingScope: "test-recipe", // triggers strict references
		IntelligentRouting: IntelligentRouting{
			Signals: Signals{
				KeywordRules: []KeywordRule{{Name: "billing"}},
				Categories:   []Category{testDomainCategory("ucla")},
			},
			Decisions: []Decision{{
				Name: "test-decision",
				Rules: RuleCombination{Conditions: []RuleNode{
					{Type: SignalTypeKeyword, Name: "billing"},
					{Type: SignalTypeDomain, Name: "ucla"},
				}},
			}},
		},
	}

	err := validateDecisionSignalReferences(cfg)
	require.NoError(t, err)
}

// TestValidateDecisionSignalReferences_AuthzNamesTheRole verifies that an
// authz condition resolves against the roles the bindings grant, which is
// what the authz signal emits at request time. A binding's own name never
// matches a request, so referencing it is an error.
func TestValidateDecisionSignalReferences_AuthzNamesTheRole(t *testing.T) {
	tests := []struct {
		name      string
		reference string
		wantError string
	}{
		{name: "role", reference: "admin"},
		{
			name:      "binding name",
			reference: "admins",
			wantError: `recipes[test-recipe].decisions["admin-route"]: signal authz("admins") is not declared in this recipe`,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			cfg := &RouterConfig{
				RoutingScope: "test-recipe", // triggers strict references
				IntelligentRouting: IntelligentRouting{
					Signals: Signals{
						RoleBindings: []RoleBinding{{
							Name:     "admins",
							Role:     "admin",
							Subjects: []Subject{{Kind: "Group", Name: "platform-admins"}},
						}},
					},
					Decisions: []Decision{{
						Name:  "admin-route",
						Rules: RuleCombination{Conditions: []RuleNode{{Type: SignalTypeAuthz, Name: tc.reference}}},
					}},
				},
			}

			err := validateDecisionSignalReferences(cfg)
			if tc.wantError == "" {
				require.NoError(t, err)
				return
			}
			require.EqualError(t, err, tc.wantError)
		})
	}
}
