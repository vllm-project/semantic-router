package postgres

import (
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementidentity"
)

type scanner interface{ Scan(...any) error }

func scanPrincipal(row scanner) (managementidentity.Principal, error) {
	var (
		principal  managementidentity.Principal
		attributes []byte
		revision   int64
	)
	err := row.Scan(
		&principal.Identity.ID, &principal.Identity.Issuer, &principal.Identity.Subject,
		&principal.DisplayName, &principal.VerifiedEmail, &attributes,
		&principal.Identity.Status, &revision, &principal.Identity.CreatedAt, &principal.Identity.UpdatedAt,
	)
	if err != nil {
		return managementidentity.Principal{}, err
	}
	if err := json.Unmarshal(attributes, &principal.Identity.Attributes); err != nil {
		return managementidentity.Principal{}, errors.New("stored Management principal attributes are invalid")
	}
	principal.Identity.CreatedAt = utc(principal.Identity.CreatedAt)
	principal.Identity.UpdatedAt = utc(principal.Identity.UpdatedAt)
	principal.Revision = accesscontrol.Revision(revision)
	if err := principal.Identity.Validate(); err != nil || principal.Revision == 0 || principal.DisplayName == "" {
		return managementidentity.Principal{}, errors.New("stored Management principal is invalid")
	}
	return principal, nil
}

func scanRole(row scanner) (managementidentity.Role, error) {
	var (
		role           managementidentity.Role
		namespace      sql.NullString
		permissions    []byte
		permissionHash []byte
		revision       int64
	)
	err := row.Scan(
		&role.Role.ID, &namespace, &role.Role.Name, &role.Role.DisplayName,
		&role.Description, &permissions, &permissionHash, &role.Role.BuiltIn,
		&role.Role.Status, &revision, &role.CreatedAt, &role.UpdatedAt,
	)
	if err != nil {
		return managementidentity.Role{}, err
	}
	if namespace.Valid {
		role.Role.NamespaceID = accesscontrol.NamespaceID(namespace.String)
	}
	set, _, err := decodePermissionSet(permissions)
	if err != nil {
		return managementidentity.Role{}, err
	}
	_, digest, err := encodePermissionSet(set)
	if err != nil || len(permissionHash) != len(digest) || string(permissionHash) != string(digest[:]) {
		return managementidentity.Role{}, errors.New("stored Management role permission digest is invalid")
	}
	role.Role.Permissions = set
	role.Role.Revision = accesscontrol.Revision(revision)
	role.CreatedAt = utc(role.CreatedAt)
	role.UpdatedAt = utc(role.UpdatedAt)
	if err := role.Role.Validate(); err != nil {
		return managementidentity.Role{}, fmt.Errorf("stored Management role is invalid: %w", err)
	}
	return role, nil
}

func scanRoleBinding(row scanner) (managementidentity.RoleBinding, error) {
	var (
		binding      managementidentity.RoleBinding
		namespace    sql.NullString
		resourceType sql.NullString
		resourceID   sql.NullString
		ceiling      []byte
		revision     int64
	)
	err := row.Scan(
		&binding.Binding.ID, &binding.Binding.PrincipalID, &binding.Binding.RoleID,
		&binding.Binding.Scope.Kind, &namespace, &resourceType, &resourceID,
		&ceiling, &binding.Binding.Status, &revision, &binding.CreatedAt, &binding.UpdatedAt,
	)
	if err != nil {
		return managementidentity.RoleBinding{}, err
	}
	if namespace.Valid {
		binding.Binding.Scope.NamespaceID = accesscontrol.NamespaceID(namespace.String)
	}
	switch binding.Binding.Scope.Kind {
	case accesscontrol.ScopeKindTeam:
		binding.Binding.Scope.TeamID = accesscontrol.TeamID(resourceID.String)
	case accesscontrol.ScopeKindUser:
		binding.Binding.Scope.UserID = accesscontrol.UserID(resourceID.String)
	case accesscontrol.ScopeKindResource:
		binding.Binding.Scope.ResourceType = accesscontrol.ScopeResourceType(resourceType.String)
		binding.Binding.Scope.ResourceID = accesscontrol.ResourceID(resourceID.String)
	}
	set, _, err := decodePermissionSet(ceiling)
	if len(ceiling) == 2 && string(ceiling) == "[]" {
		set, err = accesscontrol.NewPermissionSet()
	}
	if err != nil {
		return managementidentity.RoleBinding{}, errors.New("stored Management role-binding ceiling is invalid")
	}
	binding.Binding.DelegationCeiling = set
	binding.Binding.Revision = accesscontrol.Revision(revision)
	binding.CreatedAt = utc(binding.CreatedAt)
	binding.UpdatedAt = utc(binding.UpdatedAt)
	if err := binding.Binding.Validate(); err != nil {
		return managementidentity.RoleBinding{}, fmt.Errorf("stored Management role binding is invalid: %w", err)
	}
	return binding, nil
}

func scopeColumns(scope accesscontrol.Scope) (any, any, any, error) {
	if err := scope.Validate(); err != nil {
		return nil, nil, nil, err
	}
	var namespace, resourceType, resourceID any
	if scope.NamespaceID != "" {
		namespace = string(scope.NamespaceID)
	}
	switch scope.Kind {
	case accesscontrol.ScopeKindTeam:
		resourceID = string(scope.TeamID)
	case accesscontrol.ScopeKindUser:
		resourceID = string(scope.UserID)
	case accesscontrol.ScopeKindResource:
		resourceType = string(scope.ResourceType)
		resourceID = string(scope.ResourceID)
	}
	return namespace, resourceType, resourceID, nil
}
