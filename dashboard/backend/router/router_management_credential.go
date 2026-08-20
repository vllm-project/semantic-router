package router

import (
	"os"
	"strings"

	"github.com/vllm-project/semantic-router/dashboard/backend/routerauth"
	routerconfig "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// runtimeManagementCredential keeps Dashboard proxy authentication aligned
// with the active Router configuration. A stale persisted credential must not
// take Insights offline when management auth is explicitly disabled.
type runtimeManagementCredential struct {
	configPath string
	provider   routerauth.CredentialProvider
}

func (p runtimeManagementCredential) ManagementCredential() (string, error) {
	if parsed, err := routerconfig.Parse(p.configPath); err == nil {
		mode := strings.TrimSpace(parsed.ManagementAPI.Auth.Mode)
		if mode == "" || mode == routerconfig.ManagementAuthModeDisabled {
			return "", os.ErrNotExist
		}
	}
	if p.provider == nil {
		return "", os.ErrNotExist
	}
	return p.provider.ManagementCredential()
}
