package accesscontrol

import (
	"errors"
	"testing"
)

func TestEvaluateGrantsDenyWins(t *testing.T) {
	resource := GrantResource{Type: GrantResourceEntrypoint, ID: "entrypoint-1"}
	grants := []AccessPolicyGrant{
		{PolicyID: "policy-a", Resource: resource, Permission: GrantPermissionInvoke, Effect: GrantEffectAllow},
		{PolicyID: "policy-b", Resource: resource, Permission: GrantPermissionInvoke, Effect: GrantEffectDeny},
	}
	result := EvaluateGrants(grants, resource, GrantPermissionInvoke)
	if result.Decision != AccessDecisionDeny {
		t.Fatalf("decision = %q, want deny", result.Decision)
	}
	if len(result.Matched) != 2 {
		t.Fatalf("matched = %d, want 2", len(result.Matched))
	}
}

func TestEvaluateGrantsDefaultDeny(t *testing.T) {
	result := EvaluateGrants(nil, GrantResource{Type: GrantResourceModel, ID: "model-1"}, GrantPermissionDiscover)
	if result.Decision != AccessDecisionDeny || len(result.Matched) != 0 {
		t.Fatalf("unexpected result: %#v", result)
	}
}

func TestGrantRequiresExplicitUID(t *testing.T) {
	grant := AccessPolicyGrant{
		PolicyID:   "policy-1",
		Resource:   GrantResource{Type: GrantResourceModel, ID: "model-*"},
		Permission: GrantPermissionInvoke,
		Effect:     GrantEffectAllow,
	}
	if err := grant.Validate(); !errors.Is(err, ErrInvalid) {
		t.Fatalf("expected wildcard validation error, got %v", err)
	}
}

func TestResolveAccessBindingsPrecedence(t *testing.T) {
	key := validAccessBinding("binding-key", SubjectKindAPIKey, "key-1")
	user := validAccessBinding("binding-user", SubjectKindUser, "user-1")
	team := validAccessBinding("binding-team", SubjectKindTeam, "team-1")

	tests := []struct {
		name string
		key  []AccessPolicyBinding
		user []AccessPolicyBinding
		team []AccessPolicyBinding
		want InheritanceLayer
	}{
		{name: "key wins", key: []AccessPolicyBinding{key}, user: []AccessPolicyBinding{user}, team: []AccessPolicyBinding{team}, want: InheritanceLayerKey},
		{name: "disabled key falls back", key: []AccessPolicyBinding{disabledAccessBinding(key)}, user: []AccessPolicyBinding{user}, team: []AccessPolicyBinding{team}, want: InheritanceLayerUser},
		{name: "team fallback", team: []AccessPolicyBinding{team}, want: InheritanceLayerTeam},
		{name: "no policy means none", want: InheritanceLayerNone},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			resolved, err := ResolveAccessBindings(test.key, test.user, test.team)
			if err != nil {
				t.Fatalf("ResolveAccessBindings() error = %v", err)
			}
			if resolved.Source != test.want {
				t.Fatalf("source = %q, want %q", resolved.Source, test.want)
			}
		})
	}
}

func disabledAccessBinding(binding AccessPolicyBinding) AccessPolicyBinding {
	binding.Status = BindingStatusDisabled
	return binding
}
