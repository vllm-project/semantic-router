package auth

import (
	"sort"
)

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
