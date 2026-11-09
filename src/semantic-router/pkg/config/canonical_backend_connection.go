package config

import (
	"fmt"
	"strings"
)

// A backend in providers.models[].backend_refs connects with the same block
// as a deployment: endpoint (host:port with protocol, or a full URL), auth
// {header, prefix, api_key_env} and headers. base_url, api_key, api_key_env,
// auth_header, auth_prefix and extra_headers are the deprecated spellings;
// they are read as before, with a warning.

// normalizeBackendConnections reads every backend's auth and headers into
// the fields the provider catalog compiles.
func normalizeBackendConnections(canonical *CanonicalConfig) error {
	for i := range canonical.Providers.Models {
		model := &canonical.Providers.Models[i]
		for j := range model.BackendRefs {
			if err := model.BackendRefs[j].normalizeConnection(); err != nil {
				return fmt.Errorf("providers.models[%s].backend_refs[%d]: %w", model.Name, j, err)
			}
		}
	}
	return nil
}

func (b *CanonicalBackendRef) normalizeConnection() error {
	if b.Auth != nil {
		if b.AuthHeader != "" || b.AuthPrefix != nil || b.APIKeyEnv != "" || b.APIKey != "" {
			return fmt.Errorf("auth replaces auth_header, auth_prefix, api_key_env and api_key; set auth only")
		}
		if b.Auth.APIKeyFile != "" {
			return fmt.Errorf("auth.api_key_file applies to global.model_catalog.deployments; a backend reads its key from auth.api_key_env")
		}
		if err := b.Auth.validateHeader(); err != nil {
			return err
		}
		b.AuthHeader, b.AuthPrefix, b.APIKeyEnv = b.Auth.Header, b.Auth.Prefix, b.Auth.APIKeyEnv
		b.Auth = nil
	}
	if len(b.Headers) > 0 {
		if len(b.ExtraHeaders) > 0 {
			return fmt.Errorf("headers replaces extra_headers; set headers only")
		}
		if err := validateConnectionHeaders(b.Headers, &DeploymentAuth{Header: b.AuthHeader}); err != nil {
			return err
		}
		b.ExtraHeaders, b.Headers = b.Headers, nil
	}
	return nil
}

// canonicalConnection writes a backend's connection in the canonical shape:
// a base_url as endpoint, the credential header and its key's variable as
// auth, and extra_headers as headers. A key written inline stays where it
// was written.
func (b CanonicalBackendRef) canonicalConnection() CanonicalBackendRef {
	if base := strings.TrimSpace(b.BaseURL); base != "" {
		b.Endpoint, b.Protocol, b.BaseURL = base, "", ""
	}
	if b.APIKey == "" && (b.AuthHeader != "" || b.AuthPrefix != nil || b.APIKeyEnv != "") {
		b.Auth = &DeploymentAuth{Header: b.AuthHeader, Prefix: b.AuthPrefix, APIKeyEnv: b.APIKeyEnv}
		b.AuthHeader, b.AuthPrefix, b.APIKeyEnv = "", nil, ""
	}
	if len(b.ExtraHeaders) > 0 {
		b.Headers, b.ExtraHeaders = copyStringMap(b.ExtraHeaders), nil
	}
	return b
}

// backendConnectionWarnings names each deprecated connection field a
// backend writes, with its replacement.
func backendConnectionWarnings(models []CanonicalProviderModel) []ConfigWarning {
	var warnings []ConfigWarning
	for _, model := range models {
		for j, backend := range model.BackendRefs {
			path := fmt.Sprintf("providers.models[%s].backend_refs[%d]", model.Name, j)
			for _, field := range []struct {
				set         bool
				name, after string
			}{
				{strings.TrimSpace(backend.BaseURL) != "", "base_url", "endpoint, a full URL"},
				{backend.APIKey != "", "api_key", "auth.api_key_env; a key written in the configuration is deprecated"},
				{backend.APIKeyEnv != "", "api_key_env", "auth.api_key_env"},
				{backend.AuthHeader != "", "auth_header", "auth.header"},
				{backend.AuthPrefix != nil, "auth_prefix", "auth.prefix"},
				{len(backend.ExtraHeaders) > 0, "extra_headers", "headers"},
			} {
				if field.set {
					warnings = append(warnings, ConfigWarning{
						Code: warningDeprecatedModelField, Field: path + "." + field.name,
						Message: field.name + " is deprecated; write it as " + field.after,
					})
				}
			}
		}
	}
	return warnings
}
