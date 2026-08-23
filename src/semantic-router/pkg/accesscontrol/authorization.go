package accesscontrol

type ManagementRoleBinding struct {
	ID                ManagementRoleBindingID
	PrincipalID       ManagementPrincipalID
	RoleID            ManagementRoleID
	Scope             Scope
	DelegationCeiling PermissionSet
	Status            BindingStatus
	Revision          Revision
}

func (b ManagementRoleBinding) Validate() error {
	var statusErr error
	if !b.Status.Valid() {
		statusErr = invalid("status", "is not a valid role-binding status")
	}
	return joinValidation(
		validateRequired("id", string(b.ID)),
		validateRequired("principal_id", string(b.PrincipalID)),
		validateRequired("role_id", string(b.RoleID)),
		b.Scope.Validate(),
		b.DelegationCeiling.ValidateDelegable(),
		statusErr,
		validateRevision(b.Revision),
	)
}

func ValidateManagementRoleBindingReferences(
	binding ManagementRoleBinding,
	principal ManagementPrincipal,
	role ManagementRole,
) error {
	if err := binding.Validate(); err != nil {
		return err
	}
	if err := principal.Validate(); err != nil {
		return invalid("principal", err.Error())
	}
	if err := role.Validate(); err != nil {
		return invalid("role", err.Error())
	}
	if binding.PrincipalID != principal.ID {
		return invalid("principal_id", "does not match the bound principal")
	}
	if err := validateBindingRoleScope(binding, role); err != nil {
		return err
	}
	return nil
}

// Authorizes evaluates usable role permissions and scope. A delegation ceiling
// is intentionally ignored because it never grants runtime authority.
func Authorizes(
	binding ManagementRoleBinding,
	role ManagementRole,
	permission Permission,
	target ScopedTarget,
) (bool, error) {
	if err := binding.Validate(); err != nil {
		return false, err
	}
	if err := role.Validate(); err != nil {
		return false, err
	}
	if err := target.Validate(); err != nil {
		return false, err
	}
	if !permission.Valid() {
		return false, invalid("permission", "is not registered")
	}
	if err := validateBindingRoleScope(binding, role); err != nil {
		return false, err
	}
	if binding.Status != BindingStatusActive || role.Status != RoleStatusActive {
		return false, nil
	}
	if !role.Permissions.Contains(permission) {
		return false, nil
	}
	return binding.Scope.Contains(target), nil
}

// CanDelegateRoleBinding applies the single-source delegation rule. The same
// active source must provide role_binding.manage, contain the target scope, and
// have a ceiling containing both the target role and target ceiling.
func CanDelegateRoleBinding(
	sourceBinding ManagementRoleBinding,
	sourceRole ManagementRole,
	targetRole ManagementRole,
	targetScope ScopedTarget,
	targetCeiling PermissionSet,
) error {
	if err := validateDelegationSource(sourceBinding, sourceRole); err != nil {
		return err
	}
	return validateDelegationTarget(sourceBinding, targetRole, targetScope, targetCeiling)
}

func validateDelegationSource(sourceBinding ManagementRoleBinding, sourceRole ManagementRole) error {
	if err := sourceBinding.Validate(); err != nil {
		return err
	}
	if err := sourceRole.Validate(); err != nil {
		return invalid("source_role", err.Error())
	}
	if err := validateBindingRoleScope(sourceBinding, sourceRole); err != nil {
		return invalid("source_role", err.Error())
	}
	if sourceBinding.Status != BindingStatusActive || sourceRole.Status != RoleStatusActive {
		return invalid("source_binding", "must be active")
	}
	if !sourceRole.Permissions.Contains(PermissionRoleBindingManage) {
		return invalid("source_role", "does not grant role_binding.manage")
	}
	return nil
}

func validateDelegationTarget(
	sourceBinding ManagementRoleBinding,
	targetRole ManagementRole,
	targetScope ScopedTarget,
	targetCeiling PermissionSet,
) error {
	if err := targetRole.Validate(); err != nil {
		return invalid("target_role", err.Error())
	}
	if err := targetCeiling.ValidateDelegable(); err != nil {
		return err
	}
	if err := targetScope.Validate(); err != nil {
		return err
	}
	if !sourceBinding.Scope.Contains(targetScope) {
		return invalid("target_scope", "is not contained by the source binding")
	}
	if err := validateRoleScope(targetScope.Scope, targetRole); err != nil {
		return invalid("target_scope", err.Error())
	}
	if !sourceBinding.DelegationCeiling.ContainsAll(targetRole.Permissions) {
		return invalid("target_role", "permissions exceed the source delegation ceiling")
	}
	if !sourceBinding.DelegationCeiling.ContainsAll(targetCeiling) {
		return invalid("target_ceiling", "exceeds the source delegation ceiling")
	}
	return nil
}

func validateBindingRoleScope(binding ManagementRoleBinding, role ManagementRole) error {
	if binding.RoleID != role.ID {
		return invalid("role_id", "does not match the supplied role")
	}
	return validateRoleScope(binding.Scope, role)
}

func validateRoleScope(scope Scope, role ManagementRole) error {
	if !role.BuiltIn && (scope.Kind == ScopeKindCluster || scope.NamespaceID != role.NamespaceID) {
		return invalid("scope", "a custom role may be bound only within its owner namespace")
	}
	if role.BuiltIn && BuiltInRoleName(role.Name) == BuiltInRoleConsumer && scope.Kind != ScopeKindUser {
		return invalid("scope", "the consumer role may be bound only to a user")
	}
	return nil
}

// CanCreateCustomRole applies the custom-role creation half of the same
// non-combinable source rule.
func CanCreateCustomRole(
	sourceBinding ManagementRoleBinding,
	sourceRole ManagementRole,
	customRole ManagementRole,
) error {
	if customRole.BuiltIn {
		return invalid("custom_role", "must not be built in")
	}
	if err := customRole.Validate(); err != nil {
		return err
	}
	if err := sourceBinding.Validate(); err != nil {
		return err
	}
	if err := sourceRole.Validate(); err != nil {
		return err
	}
	if sourceBinding.RoleID != sourceRole.ID || sourceBinding.Status != BindingStatusActive || sourceRole.Status != RoleStatusActive {
		return invalid("source_binding", "must be active and match the source role")
	}
	if !sourceRole.Permissions.Contains(PermissionManagementRoleManage) {
		return invalid("source_role", "does not grant management_role.manage")
	}
	target := ScopedTarget{Scope: NamespaceScope(customRole.NamespaceID)}
	if !sourceBinding.Scope.Contains(target) {
		return invalid("namespace_id", "is outside the source binding scope")
	}
	if !sourceBinding.DelegationCeiling.ContainsAll(customRole.Permissions) {
		return invalid("permissions", "exceed the source delegation ceiling")
	}
	return nil
}
