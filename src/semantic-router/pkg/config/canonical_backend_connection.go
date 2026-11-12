package config

import (
	"fmt"

	modelcatalog "github.com/vllm-project/semantic-router/src/semantic-router/pkg/catalog"
)

// A backend in providers.models[].backend_refs connects with the same block
// as a deployment: endpoint (host:port with protocol, or a full URL), auth
// {credential, header, prefix, api_key_env}, headers and tls. A key the
// configuration writes inline (api_key) goes in the auth header as an
// api_key_env key would.

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
		if b.Auth.APIKeyFile != "" {
			return fmt.Errorf("auth.api_key_file applies to global.model_catalog.deployments; a backend reads its key from auth.api_key_env")
		}
		if b.Auth.APIKeyEnv != "" && b.APIKey != "" {
			return fmt.Errorf("auth.api_key_env and api_key both name the key; set one")
		}
		if err := b.Auth.validateHeader(); err != nil {
			return err
		}
		if err := b.Auth.validateCredential("api_key or auth.api_key_env", b.APIKey != "" || b.Auth.APIKeyEnv != ""); err != nil {
			return err
		}
	}
	if err := b.validateTLS(builtIn); err != nil {
		return err
	}
	return validateConnectionHeaders(b.Headers, b.Auth)
}
