package config

import (
	"slices"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
)

// HeaderNames lists the headers that carry a caller's identity: the x-authz-*
// set an authenticator in front of the Router sets, and the names this
// configuration reads the user and groups from. Only that authenticator may
// set them, and no model backend receives them.
func (ic IdentityConfig) HeaderNames() []string {
	names := []string{headers.AuthzUserID, headers.AuthzUserGroups, headers.AuthzTeamID, headers.AuthzTenantID}
	for _, name := range []string{ic.GetUserIDHeader(), ic.GetUserGroupsHeader()} {
		name = strings.ToLower(strings.TrimSpace(name))
		if name != "" && !slices.Contains(names, name) {
			names = append(names, name)
		}
	}
	return names
}
