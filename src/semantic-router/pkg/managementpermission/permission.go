// Package managementpermission defines the dependency-neutral authorization
// expression contract shared by the Management API registry and evaluator.
package managementpermission

import (
	"fmt"
	"sort"
	"strings"
)

type Operator string

const (
	Leaf        Operator = "permission"
	All         Operator = "all"
	Any         Operator = "any"
	Conditional Operator = "conditional"
	Special     Operator = "special_authentication"
	Recorded    Operator = "recorded_operation_permission"
)

// Expression is a serializable authorization AST. Permission leaves name
// both the permission and scope operand; conditional nodes preserve
// request-dependent conjunctions rather than weakening them into prose.
type Expression struct {
	Operator   Operator     `json:"operator"`
	Permission string       `json:"permission,omitempty"`
	Scope      string       `json:"scope,omitempty"`
	Condition  string       `json:"condition,omitempty"`
	Mechanism  string       `json:"mechanism,omitempty"`
	Reference  string       `json:"reference,omitempty"`
	Operands   []Expression `json:"operands,omitempty"`
}

func Require(permission, scope string) Expression {
	return Expression{Operator: Leaf, Permission: permission, Scope: scope}
}

func RequireAll(operands ...Expression) Expression {
	return Expression{Operator: All, Operands: operands}
}

func RequireAny(operands ...Expression) Expression {
	return Expression{Operator: Any, Operands: operands}
}

func RequireWhen(condition string, operand Expression) Expression {
	return Expression{Operator: Conditional, Condition: condition, Operands: []Expression{operand}}
}

func RequireSpecial(mechanism string) Expression {
	return Expression{Operator: Special, Mechanism: mechanism}
}

func RequireRecorded(reference string) Expression {
	return Expression{Operator: Recorded, Reference: reference}
}

func (e Expression) Validate() error {
	switch e.Operator {
	case Leaf:
		if !knownPermissions[e.Permission] {
			return fmt.Errorf("unknown permission %q", e.Permission)
		}
		if !knownPermissionScopes[e.Scope] {
			return fmt.Errorf("unknown permission scope %q", e.Scope)
		}
		if len(e.Operands) != 0 || e.Condition != "" || e.Mechanism != "" || e.Reference != "" {
			return fmt.Errorf("permission leaf %q contains non-leaf fields", e.Permission)
		}
	case All, Any:
		if len(e.Operands) < 2 {
			return fmt.Errorf("%s expression requires at least two operands", e.Operator)
		}
		if e.Permission != "" || e.Scope != "" || e.Condition != "" || e.Mechanism != "" || e.Reference != "" {
			return fmt.Errorf("%s expression contains leaf fields", e.Operator)
		}
		for i := range e.Operands {
			if err := e.Operands[i].Validate(); err != nil {
				return fmt.Errorf("%s operand %d: %w", e.Operator, i, err)
			}
		}
	case Conditional:
		if !knownPermissionConditions[e.Condition] {
			return fmt.Errorf("unknown permission condition %q", e.Condition)
		}
		if len(e.Operands) != 1 {
			return fmt.Errorf("conditional expression requires exactly one operand")
		}
		if e.Permission != "" || e.Scope != "" || e.Mechanism != "" || e.Reference != "" {
			return fmt.Errorf("conditional expression contains unrelated fields")
		}
		if err := e.Operands[0].Validate(); err != nil {
			return fmt.Errorf("conditional operand: %w", err)
		}
	case Special:
		if !knownAuthenticationMechanisms[e.Mechanism] {
			return fmt.Errorf("unknown special authentication mechanism %q", e.Mechanism)
		}
		if len(e.Operands) != 0 || e.Permission != "" || e.Scope != "" || e.Condition != "" || e.Reference != "" {
			return fmt.Errorf("special authentication expression contains unrelated fields")
		}
	case Recorded:
		if !knownRecordedPermissionReferences[e.Reference] {
			return fmt.Errorf("unknown recorded permission reference %q", e.Reference)
		}
		if len(e.Operands) != 0 || e.Permission != "" || e.Scope != "" || e.Condition != "" || e.Mechanism != "" {
			return fmt.Errorf("recorded permission expression contains unrelated fields")
		}
	default:
		return fmt.Errorf("unknown permission operator %q", e.Operator)
	}
	return nil
}

func (e Expression) Canonical() string {
	switch e.Operator {
	case Leaf:
		return e.Permission + "@" + e.Scope
	case Special:
		return "AUTH(" + e.Mechanism + ")"
	case Recorded:
		return "RECORDED(" + e.Reference + ")"
	case Conditional:
		if len(e.Operands) != 1 {
			return ""
		}
		return "WHEN(" + e.Condition + "," + e.Operands[0].Canonical() + ")"
	case All, Any:
		parts := make([]string, len(e.Operands))
		for i := range e.Operands {
			parts[i] = e.Operands[i].Canonical()
		}
		separator := " AND "
		if e.Operator == Any {
			separator = " OR "
		}
		return "(" + strings.Join(parts, separator) + ")"
	default:
		return ""
	}
}

func RegisteredPermissions() []string {
	permissions := make([]string, 0, len(knownPermissions))
	for permission := range knownPermissions {
		permissions = append(permissions, permission)
	}
	sort.Strings(permissions)
	return permissions
}

var knownPermissions = stringSet(
	"self.read", "self.manage",
	"cluster.read", "cluster.manage",
	"namespace.read", "namespace.manage",
	"identity_issuer.read", "identity_issuer.manage",
	"principal.read", "principal.manage",
	"principal_directory.read", "principal_link.read", "principal_link.manage",
	"management_role.read", "management_role.manage",
	"role_binding.read", "role_binding.manage",
	"service_account.read", "service_account.manage",
	"invitation.read", "invitation.manage", "onboarding.manage",
	"user.read", "user.manage",
	"team.read", "team.manage", "membership.manage",
	"key.read", "key.manage", "key.reveal",
	"delegation.use", "delegation.manage",
	"access_policy.read", "access_policy.manage",
	"rate_policy.read", "rate_policy.manage",
	"routing_context.read", "routing_context.manage",
	"routing.read", "routing.manage", "routing.publish", "evaluation.run",
	"agent.read", "agent.use", "agent.manage",
	"tool.read", "tool.invoke", "tool.manage",
	"provider_catalog.read",
	"provider_credential.read", "provider_credential.manage", "provider_credential.use",
	"quota.read", "quota.reconcile",
	"usage.read", "usage.internal_dimensions.read",
	"log.read", "log_payload.read", "audit.read",
	"operation.read", "operation.manage", "health.read",
)

var knownPermissionScopes = stringSet(
	"intrinsic_self",
	"cluster",
	"path_namespace",
	"request_namespace",
	"target",
	"owner",
	"current_owner",
	"target_owner",
	"subject",
	"resource",
	"policy",
	"access_policy",
	"rate_policy",
	"current_policy_defaults",
	"target_policy_defaults",
	"current_access_policy_default",
	"current_rate_policy_default",
	"target_access_policy_default",
	"target_rate_policy_default",
	"team",
	"user",
	"key",
	"credential",
	"all_returned_resources",
	"all_returned_bindings",
	"all_affected_bindings",
	"all_dependencies",
	"attributed_subject",
	"operation_origin",
	"operation_targets",
)

var knownPermissionConditions = stringSet(
	"access_policy_binding_requested",
	"rate_policy_binding_requested",
	"inline_rate_policy_requested",
	"current_access_policy_default_present",
	"current_rate_policy_default_present",
	"target_access_policy_default_present",
	"target_rate_policy_default_present",
	"team_role_requested",
	"team_membership_requested",
	"first_key_requested",
	"role_binding_requested",
	"access_binding_requested",
	"rate_binding_requested",
	"user_owner",
	"team_owner",
	"key_owner",
	"current_user_owner",
	"current_team_owner",
	"target_user_owner",
	"target_team_owner",
	"access_policy_references_routing_resources",
	"routing_context_override_requested",
	"routing_subject_supplied",
	"provider_credential_supplied",
	"provider_credential_referenced",
	"no_provider_credential_supplied",
	"request_log_payload_requested",
	"internal_usage_dimensions_requested",
	"fence_actual_reconciliation",
	"fence_payload_evidence_requested",
	"fence_actor_or_audit_fields_requested",
	"cross_actor_operation",
	"operation_originator",
	"operation_cancel",
	"secret_result_claim",
	"sensitive_action",
	"namespace_list_item",
	"user_membership_row",
	"entrypoint_topology_requested",
	"entrypoint_resolution_matched",
)

var knownAuthenticationMechanisms = stringSet(
	"bootstrap_credential",
	"recovery_credential",
	"exchange_challenge",
	"subject_token_exchange",
	"service_credential_or_mtls",
	"trusted_issuer_logout_token",
	"onboarding_secret_claim_capability",
)

var knownRecordedPermissionReferences = stringSet(
	"original_domain_read",
	"original_domain_mutation",
	"original_secret_permission",
)

func stringSet(values ...string) map[string]bool {
	result := make(map[string]bool, len(values))
	for _, value := range values {
		result[value] = true
	}
	return result
}
