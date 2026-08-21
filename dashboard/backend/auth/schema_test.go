package auth

import (
	"testing"
)

func TestNewPermissionsExistInAllPermissions(t *testing.T) {
	t.Parallel()

	requiredPerms := []string{PermFeedbackSubmit, PermReplayRead, PermAccessRead, PermAccessManage, PermUsageRead, PermAuditRead}
	allSet := make(map[string]bool, len(AllPermissions))
	for _, p := range AllPermissions {
		allSet[p] = true
	}

	for _, perm := range requiredPerms {
		if !allSet[perm] {
			t.Fatalf("permission %q missing from AllPermissions", perm)
		}
	}
}

func TestRuntimeLogsRequireWriteOrAdminRoleByDefault(t *testing.T) {
	t.Parallel()

	for _, role := range []string{RoleAdmin, RoleWrite} {
		if !containsPermission(DefaultRolePermissions[role], PermLogsRead) {
			t.Fatalf("role %q should have %q permission", role, PermLogsRead)
		}
	}
	if containsPermission(DefaultRolePermissions[RoleRead], PermLogsRead) {
		t.Fatalf("read role should not have %q permission", PermLogsRead)
	}
}

func containsPermission(permissions []string, target string) bool {
	for _, permission := range permissions {
		if permission == target {
			return true
		}
	}
	return false
}

func TestReadOnlyUsersReceiveScopedAnalysisWithoutOperatorPermissions(t *testing.T) {
	t.Parallel()

	for _, role := range SupportedRoles {
		perms := DefaultRolePermissions[role]
		hasFeedback := false
		hasReplay := false
		for _, p := range perms {
			if p == PermFeedbackSubmit {
				hasFeedback = true
			}
			if p == PermReplayRead {
				hasReplay = true
			}
		}
		if hasFeedback != (role != RoleRead) {
			t.Fatalf("role %q feedback permission = %v", role, hasFeedback)
		}
		if !hasReplay {
			t.Fatalf("role %q replay permission = %v", role, hasReplay)
		}
	}
	if !containsPermission(DefaultRolePermissions[RoleRead], PermEvalRead) {
		t.Fatalf("read role should have scoped evaluation access")
	}
	for _, permission := range []string{PermOpenClawRead, PermMcpRead, PermStatusRead} {
		if containsPermission(DefaultRolePermissions[RoleRead], permission) {
			t.Fatalf("read role should not have operate permission %q", permission)
		}
	}
}

func TestDefaultRolePermissionsCoversAllSupportedRoles(t *testing.T) {
	t.Parallel()

	for _, role := range SupportedRoles {
		if _, ok := DefaultRolePermissions[role]; !ok {
			t.Fatalf("role %q missing from DefaultRolePermissions", role)
		}
	}
}
