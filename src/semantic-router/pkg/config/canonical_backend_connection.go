package config

import (
	"fmt"
	"path/filepath"

	modelcatalog "github.com/vllm-project/semantic-router/src/semantic-router/pkg/catalog"
)

// A backend in providers.models[].backend_refs connects with the same block
// as a deployment: endpoint (host:port with protocol, or a full URL), auth
// {credential, header, prefix, api_key_env, api_key_file}, headers and tls.
// A key the configuration writes inline (api_key) goes in the auth header as
// an api_key_env or api_key_file key would.

// validateBackendConnections checks every backend's auth, headers and tls,
// and the default CA of https backends.
func validateBackendConnections(canonical *CanonicalConfig) error {
	if err := canonical.Providers.Defaults.TLS.validate(); err != nil {
		return fmt.Errorf("providers.defaults.%w", err)
	}
	builtIn, err := modelcatalog.BuiltIn()
	if err != nil {
		return fmt.Errorf("load built-in model catalog: %w", err)
	}
	for _, model := range canonical.Providers.Models {
		for j, backend := range model.BackendRefs {
			if err := backend.validateConnection(builtIn); err != nil {
				return fmt.Errorf("providers.models[%s].backend_refs[%d]: %w", model.Name, j, err)
			}
		}
	}
	return nil
}

func (b CanonicalBackendRef) validateConnection(builtIn *modelcatalog.Registry) error {
	if b.Auth != nil {
		if err := b.validateKeySource(); err != nil {
			return err
		}
		if err := b.Auth.validateHeader(); err != nil {
			return err
		}
		keys := b.APIKey != "" || b.Auth.APIKeyEnv != "" || b.Auth.APIKeyFile != ""
		if err := b.Auth.validateCredential("api_key, auth.api_key_env or auth.api_key_file", keys); err != nil {
			return err
		}
	}
	if err := b.validateTLS(builtIn); err != nil {
		return err
	}
	return validateConnectionHeaders(b.Headers, b.Auth)
}

// validateKeySource allows one key source: api_key, auth.api_key_env, or an
// absolute auth.api_key_file the Router reads at every load.
func (b CanonicalBackendRef) validateKeySource() error {
	sources := 0
	for _, set := range []bool{b.APIKey != "", b.Auth.APIKeyEnv != "", b.Auth.APIKeyFile != ""} {
		if set {
			sources++
		}
	}
	if sources > 1 {
		return fmt.Errorf("api_key, auth.api_key_env and auth.api_key_file each name the key; set one")
	}
	if b.Auth.APIKeyFile != "" && !filepath.IsAbs(b.Auth.APIKeyFile) {
		return fmt.Errorf("auth.api_key_file must be an absolute path")
	}
	return nil
}
