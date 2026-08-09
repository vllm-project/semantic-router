package auth

import (
	"testing"
)

func TestNewPermissionsExistInAllPermissions(t *testing.T) {
	t.Parallel()

	requiredPerms := []string{PermFeedbackSubmit, PermReplayRead}
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

// TestRetiredPermissionsAbsentFromRoleMetadata pins the removal of the dashboard security
// policy surface. A retired key left in role metadata would be handed back out by
// syncDefaultRolePermissions on every start and shown as grantable in the admin UI.
func TestRetiredPermissionsAbsentFromRoleMetadata(t *testing.T) {
	t.Parallel()

	for _, retired := range retiredPermissions {
		for _, p := range AllPermissions {
			if p == retired {
				t.Fatalf("retired permission %q is still advertised in AllPermissions", retired)
			}
		}
		for role, perms := range DefaultRolePermissions {
			for _, p := range perms {
				if p == retired {
					t.Fatalf("role %q still grants retired permission %q", role, retired)
				}
			}
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
