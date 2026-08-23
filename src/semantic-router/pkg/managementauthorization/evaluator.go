package managementauthorization

import (
	"errors"
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementpermission"
)

var (
	// ErrDenied intentionally carries no resource details. HTTP middleware maps
	// it to the operation's nondisclosing 403/404 contract.
	ErrDenied = errors.New("management operation is not authorized")
	// ErrInvalidContext identifies a server integration error. It must fail
	// closed and must never be returned to an untrusted caller verbatim.
	ErrInvalidContext = errors.New("invalid management authorization context")
)

// RoleGrant is an authoritative active-or-inactive binding/role pair loaded
// for the authenticated Management principal. Authorizes performs lifecycle,
// role identity, permission, and scope checks for every leaf.
type RoleGrant struct {
	Binding accesscontrol.ManagementRoleBinding
	Role    accesscontrol.ManagementRole
}

// TeamGrant carries a live Team membership and namespace self-service options.
// The evaluator synthesizes the fixed, non-delegable permission set itself so
// callers cannot accidentally turn a Team membership into an arbitrary role.
type TeamGrant struct {
	Membership accesscontrol.TeamMembership
	Options    accesscontrol.TeamEntitlementOptions
}

// EvaluationContext contains only trusted facts. Targets are keyed by the
// scope operands registered in managementapi. A leaf over multiple targets is
// satisfied only when every target is covered; list and bulk authorization can
// therefore never leak a partially authorized result.
type EvaluationContext struct {
	Authenticated bool
	RoleGrants    []RoleGrant
	TeamGrants    []TeamGrant
	Targets       map[string][]accesscontrol.ScopedTarget
	Conditions    map[string]bool
	SpecialAuth   map[string]bool
	Recorded      map[string]bool
}

// Evaluate applies the exact permission AST published in OpenAPI. It returns
// ErrDenied for a well-formed but unsatisfied expression and wraps
// ErrInvalidContext for malformed trusted inputs.
func Evaluate(expression managementpermission.Expression, context EvaluationContext) error {
	if err := expression.Validate(); err != nil {
		return fmt.Errorf("%w: permission expression: %w", ErrInvalidContext, err)
	}
	if err := validateContext(context); err != nil {
		return err
	}
	allowed, err := evaluate(expression, context)
	if err != nil {
		return err
	}
	if !allowed {
		return ErrDenied
	}
	return nil
}

func evaluate(expression managementpermission.Expression, context EvaluationContext) (bool, error) {
	switch expression.Operator {
	case managementpermission.Leaf:
		return evaluateLeaf(expression, context)
	case managementpermission.All:
		for _, operand := range expression.Operands {
			allowed, err := evaluate(operand, context)
			if err != nil || !allowed {
				return allowed, err
			}
		}
		return true, nil
	case managementpermission.Any:
		for _, operand := range expression.Operands {
			allowed, err := evaluate(operand, context)
			if err != nil {
				return false, err
			}
			if allowed {
				return true, nil
			}
		}
		return false, nil
	case managementpermission.Conditional:
		// A false condition makes the conditional requirement inapplicable.
		if !context.Conditions[expression.Condition] {
			return true, nil
		}
		return evaluate(expression.Operands[0], context)
	case managementpermission.Special:
		return context.SpecialAuth[expression.Mechanism], nil
	case managementpermission.Recorded:
		return context.Recorded[expression.Reference], nil
	default:
		return false, fmt.Errorf("%w: unsupported operator %q", ErrInvalidContext, expression.Operator)
	}
}

func evaluateLeaf(expression managementpermission.Expression, context EvaluationContext) (bool, error) {
	permission := accesscontrol.Permission(expression.Permission)
	if !permission.Valid() {
		return false, fmt.Errorf("%w: unknown permission %q", ErrInvalidContext, expression.Permission)
	}
	if expression.Scope == "intrinsic_self" {
		return context.Authenticated && permission.Intrinsic(), nil
	}
	if permission.Intrinsic() {
		return false, fmt.Errorf("%w: intrinsic permission %q used with scope %q", ErrInvalidContext, permission, expression.Scope)
	}
	targets, exists := context.Targets[expression.Scope]
	if !exists || len(targets) == 0 {
		return false, fmt.Errorf("%w: unresolved scope operand %q", ErrInvalidContext, expression.Scope)
	}
	for _, target := range targets {
		covered, err := targetCovered(permission, target, context)
		if err != nil {
			return false, err
		}
		if !covered {
			return false, nil
		}
	}
	return true, nil
}

func targetCovered(permission accesscontrol.Permission, target accesscontrol.ScopedTarget, context EvaluationContext) (bool, error) {
	for _, grant := range context.RoleGrants {
		allowed, err := accesscontrol.Authorizes(grant.Binding, grant.Role, permission, target)
		if err != nil {
			return false, fmt.Errorf("%w: role grant: %w", ErrInvalidContext, err)
		}
		if allowed {
			return true, nil
		}
	}
	for _, grant := range context.TeamGrants {
		if grant.Membership.Status != accesscontrol.MembershipStatusActive {
			continue
		}
		permissions, err := accesscontrol.TeamRolePermissions(grant.Membership.Role, grant.Options)
		if err != nil {
			return false, fmt.Errorf("%w: team grant: %w", ErrInvalidContext, err)
		}
		scope := accesscontrol.TeamScope(grant.Membership.NamespaceID, grant.Membership.TeamID)
		if permissions.Contains(permission) && scope.Contains(target) {
			return true, nil
		}
	}
	return false, nil
}

func validateContext(context EvaluationContext) error {
	for scopeOperand, targets := range context.Targets {
		if scopeOperand == "" || len(targets) == 0 {
			return fmt.Errorf("%w: empty target set", ErrInvalidContext)
		}
		for _, target := range targets {
			if err := target.Validate(); err != nil {
				return fmt.Errorf("%w: target %q: %w", ErrInvalidContext, scopeOperand, err)
			}
		}
	}
	for _, grant := range context.TeamGrants {
		if err := grant.Membership.Validate(); err != nil {
			return fmt.Errorf("%w: team membership: %w", ErrInvalidContext, err)
		}
	}
	return nil
}
