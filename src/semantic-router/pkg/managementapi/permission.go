package managementapi

import "github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementpermission"

type PermissionOperator = managementpermission.Operator

const (
	PermissionLeaf        = managementpermission.Leaf
	PermissionAll         = managementpermission.All
	PermissionAny         = managementpermission.Any
	PermissionConditional = managementpermission.Conditional
	PermissionSpecial     = managementpermission.Special
	PermissionRecorded    = managementpermission.Recorded
)

type PermissionExpression = managementpermission.Expression

func Require(permission, scope string) PermissionExpression {
	return managementpermission.Require(permission, scope)
}

func RequireAll(operands ...PermissionExpression) PermissionExpression {
	return managementpermission.RequireAll(operands...)
}

func RequireAny(operands ...PermissionExpression) PermissionExpression {
	return managementpermission.RequireAny(operands...)
}

func RequireWhen(condition string, operand PermissionExpression) PermissionExpression {
	return managementpermission.RequireWhen(condition, operand)
}

func RequireSpecial(mechanism string) PermissionExpression {
	return managementpermission.RequireSpecial(mechanism)
}

func RequireRecordedPermission(reference string) PermissionExpression {
	return managementpermission.RequireRecorded(reference)
}

func RegisteredPermissions() []string {
	return managementpermission.RegisteredPermissions()
}
