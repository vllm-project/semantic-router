package auth

import (
	"testing"
)

func TestNewPermissionsExistInAllPermissions(t *testing.T) {
	t.Parallel()

	requiredPerms := []string{PermFeedbackSubmit, PermReplayRead, PermSecurityManage}
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

func TestAdminRoleHasSecurityManage(t *testing.T) {
	t.Parallel()

	adminPerms := DefaultRolePermissions[RoleAdmin]
	found := false
	for _, p := range adminPerms {
		if p == PermSecurityManage {
			found = true
			break
		}
	}
	if !found {
		t.Fatalf("admin role should have %q permission", PermSecurityManage)
	}
}

func TestWriteRoleDoesNotHaveSecurityManage(t *testing.T) {
	t.Parallel()

	writePerms := DefaultRolePermissions[RoleWrite]
	for _, p := range writePerms {
		if p == PermSecurityManage {
			t.Fatalf("write role should not have %q permission", PermSecurityManage)
		}
	}
}

func TestReadRoleDoesNotHaveSecurityManage(t *testing.T) {
	t.Parallel()

	readPerms := DefaultRolePermissions[RoleRead]
	for _, p := range readPerms {
		if p == PermSecurityManage {
			t.Fatalf("read role should not have %q permission", PermSecurityManage)
		}
	}
}

func TestAllRolesHaveFeedbackSubmitAndReplayRead(t *testing.T) {
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
		if !hasFeedback {
			t.Fatalf("role %q should have %q permission", role, PermFeedbackSubmit)
		}
		if !hasReplay {
			t.Fatalf("role %q should have %q permission", role, PermReplayRead)
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
