package managementauthorization

import (
	"errors"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementpermission"
)

func TestEvaluateRequiresEveryTargetAndEveryConjunct(t *testing.T) {
	role, ok := accesscontrol.BuiltInRole(accesscontrol.BuiltInRoleAnalyst)
	if !ok {
		t.Fatal("analyst role missing")
	}
	namespaceID := accesscontrol.NamespaceID("ns-1")
	binding := accesscontrol.ManagementRoleBinding{
		ID: "binding-1", PrincipalID: "principal-1", RoleID: role.ID,
		Scope: accesscontrol.NamespaceScope(namespaceID), Status: accesscontrol.BindingStatusActive, Revision: 1,
	}
	context := EvaluationContext{
		Authenticated: true,
		RoleGrants:    []RoleGrant{{Binding: binding, Role: role}},
		Targets: map[string][]accesscontrol.ScopedTarget{
			"key": {{Scope: accesscontrol.ResourceScope(namespaceID, accesscontrol.ScopeResourceAPIKey, "key-1")}},
			"all_returned_bindings": {
				{Scope: accesscontrol.ResourceScope(namespaceID, accesscontrol.ScopeResourceRateLimitBinding, "binding-1")},
				{Scope: accesscontrol.ResourceScope(namespaceID, accesscontrol.ScopeResourceRateLimitBinding, "binding-2")},
			},
		},
	}
	expression := managementpermission.RequireAll(
		managementpermission.Require("key.read", "key"),
		managementpermission.Require("quota.read", "all_returned_bindings"),
	)
	if err := Evaluate(expression, context); err != nil {
		t.Fatalf("Evaluate() error = %v", err)
	}

	context.Targets["all_returned_bindings"] = append(context.Targets["all_returned_bindings"],
		accesscontrol.ScopedTarget{Scope: accesscontrol.ResourceScope("ns-2", accesscontrol.ScopeResourceRateLimitBinding, "binding-3")},
	)
	if err := Evaluate(expression, context); !errors.Is(err, ErrDenied) {
		t.Fatalf("Evaluate() error = %v, want ErrDenied", err)
	}
}

func TestEvaluateConditionAnyAndIntrinsicSelf(t *testing.T) {
	context := EvaluationContext{
		Authenticated: true,
		Conditions:    map[string]bool{"sensitive_action": false},
		SpecialAuth:   map[string]bool{"exchange_challenge": false},
	}
	expression := managementpermission.RequireAll(
		managementpermission.Require("self.read", "intrinsic_self"),
		managementpermission.RequireWhen("sensitive_action", managementpermission.RequireSpecial("exchange_challenge")),
	)
	if err := Evaluate(expression, context); err != nil {
		t.Fatalf("false conditional must be inapplicable: %v", err)
	}
	context.Conditions["sensitive_action"] = true
	if err := Evaluate(expression, context); !errors.Is(err, ErrDenied) {
		t.Fatalf("missing challenge error = %v, want ErrDenied", err)
	}
	context.SpecialAuth["exchange_challenge"] = true
	if err := Evaluate(expression, context); err != nil {
		t.Fatalf("satisfied challenge error = %v", err)
	}
}

func TestEvaluateFailsClosedOnUnresolvedScope(t *testing.T) {
	err := Evaluate(managementpermission.Require("key.read", "key"), EvaluationContext{Authenticated: true})
	if !errors.Is(err, ErrInvalidContext) {
		t.Fatalf("error = %v, want ErrInvalidContext", err)
	}
}

func TestTeamGrantCannotEscapeTeamScope(t *testing.T) {
	now := time.Unix(1, 0).UTC()
	context := EvaluationContext{
		Authenticated: true,
		TeamGrants: []TeamGrant{{
			Membership: accesscontrol.TeamMembership{
				NamespaceID: "ns-1", TeamID: "team-1", UserID: "user-1",
				Role: accesscontrol.TeamRoleMember, Status: accesscontrol.MembershipStatusActive,
				CreatedAt: now, UpdatedAt: now,
			},
		}},
		Targets: map[string][]accesscontrol.ScopedTarget{
			"team": {{Scope: accesscontrol.TeamScope("ns-1", "team-2")}},
		},
	}
	if err := Evaluate(managementpermission.Require("team.read", "team"), context); !errors.Is(err, ErrDenied) {
		t.Fatalf("error = %v, want ErrDenied", err)
	}
}
