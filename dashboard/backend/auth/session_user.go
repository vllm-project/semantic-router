package auth

import (
	"context"
	"sort"
)

func userHasPermission(ctx context.Context, svc *Service, userID, role, status, permission string) (bool, error) {
	if status != defaultUserStatusActive {
		return false, nil
	}

	perms, err := svc.store.GetEffectivePermissions(ctx, role, userID)
	if err != nil {
		return false, err
	}
	return perms[permission], nil
}

func cloneSessionUser(user *User, perms map[string]bool) *User {
	if user == nil {
		return nil
	}

	sessionUser := *user
	if len(perms) == 0 {
		sessionUser.Permissions = nil
		return &sessionUser
	}

	keys := make([]string, 0, len(perms))
	for key, allowed := range perms {
		if allowed {
			keys = append(keys, key)
		}
	}
	sort.Strings(keys)
	sessionUser.Permissions = keys
	return &sessionUser
}
