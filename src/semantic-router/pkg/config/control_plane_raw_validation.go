package config

import (
	"fmt"
	"sort"
	"strings"
)

// rejectControlPlaneSecretLiterals runs on the raw document before environment
// expansion and typed decoding. The canonical loader normally warns on unknown
// fields; secret-looking literal fields are different and must fail closed so a
// typo cannot silently turn a reference into ignored plaintext.
func rejectControlPlaneSecretLiterals(raw map[string]interface{}) error {
	global := nestedStringMap(raw["global"])
	if len(global) == 0 {
		return nil
	}
	stores := nestedStringMap(global["stores"])
	services := nestedStringMap(global["services"])

	checks := []literalFieldCheck{
		{path: "global.stores.access.postgres", node: nestedMapAt(stores, "access", "postgres"), fields: []string{"dsn", "database_url", "connection_string", "password"}},
		{path: "global.stores.access_runtime.redis", node: nestedMapAt(stores, "access_runtime", "redis"), fields: []string{"url", "redis_url", "password"}},
		{path: "global.services.access.credentials", node: nestedMapAt(services, "access", "credentials"), fields: []string{"api_key_hmac_keyring", "delegation_hmac_keyring"}},
		{path: "global.services.access.credentials.reveal", node: nestedMapAt(services, "access", "credentials", "reveal"), fields: []string{"kek_keyring"}},
		{path: "global.services.access.tenant_context", node: nestedMapAt(services, "access", "tenant_context"), fields: []string{"signing_key"}},
		{path: "global.services.backend_credentials", node: nestedStringMap(services["backend_credentials"]), fields: []string{"provider_kek_keyring"}},
		{path: "global.services.backend_egress", node: nestedStringMap(services["backend_egress"]), fields: []string{"policy", "policy_yaml"}},
		{path: "global.services.management_api.tls", node: nestedMapAt(services, "management_api", "tls"), fields: []string{"certificate", "private_key", "client_ca_bundle"}},
		{path: "global.services.management_api.auth", node: nestedMapAt(services, "management_api", "auth"), fields: []string{"token_signing_keyring", "service_account_hmac_keyring", "invitation_hmac_keyring", "control_plane_hmac_keyring", "response_kek_keyring"}},
		{path: "global.services.management_api.auth.bootstrap", node: nestedMapAt(services, "management_api", "auth", "bootstrap"), fields: []string{"token"}},
		{path: "global.services.management_api.auth.recovery", node: nestedMapAt(services, "management_api", "auth", "recovery"), fields: []string{"token"}},
	}
	for _, check := range checks {
		if err := check.reject(); err != nil {
			return err
		}
	}

	backendCredentials := nestedStringMap(services["backend_credentials"])
	reserved := map[string]bool{
		"provider_kek_keyring_file": true,
		"provider_kek_keyring_env":  true,
	}
	providerNames := make([]string, 0, len(backendCredentials))
	for name := range backendCredentials {
		if !reserved[name] {
			providerNames = append(providerNames, name)
		}
	}
	sort.Strings(providerNames)
	for _, name := range providerNames {
		if err := (literalFieldCheck{
			path:   "global.services.backend_credentials." + name,
			node:   nestedStringMap(backendCredentials[name]),
			fields: []string{"secret", "api_key", "token"},
		}).reject(); err != nil {
			return err
		}
	}
	return nil
}

type literalFieldCheck struct {
	path   string
	node   map[string]interface{}
	fields []string
}

func (c literalFieldCheck) reject() error {
	for _, field := range c.fields {
		if _, found := c.node[field]; found {
			return fmt.Errorf("%s.%s is forbidden; use the corresponding _file or _env reference", c.path, field)
		}
	}
	return nil
}

func nestedMapAt(root map[string]interface{}, path ...string) map[string]interface{} {
	current := root
	for _, segment := range path {
		if len(current) == 0 {
			return nil
		}
		current = nestedStringMap(current[segment])
	}
	return current
}

func validateBootstrapFieldNames(raw map[string]interface{}) error {
	global := nestedStringMap(raw["global"])
	if len(global) == 0 {
		return nil
	}
	checks := []struct {
		path    string
		node    map[string]interface{}
		allowed []string
	}{
		{"global.control_plane", nestedStringMap(global["control_plane"]), []string{"mode", "public_namespace_id", "provider_catalog"}},
		{"global.control_plane.provider_catalog", nestedMapAt(global, "control_plane", "provider_catalog"), []string{"replica_id_env", "lease", "renew_interval", "rollout_groups", "required_rollout_groups"}},
		{"global.stores.access", nestedMapAt(global, "stores", "access"), []string{"type", "postgres"}},
		{"global.stores.access.postgres", nestedMapAt(global, "stores", "access", "postgres"), []string{"dsn_file", "dsn_env", "max_connections"}},
		{"global.stores.access_runtime", nestedMapAt(global, "stores", "access_runtime"), []string{"type", "redis"}},
		{"global.stores.access_runtime.redis", nestedMapAt(global, "stores", "access_runtime", "redis"), []string{"url_file", "url_env", "key_prefix"}},
		{"global.services.access", nestedMapAt(global, "services", "access"), []string{"enabled", "credentials", "tenant_context", "enforcement", "usage_storage"}},
		{"global.services.access.credentials", nestedMapAt(global, "services", "access", "credentials"), []string{"api_key_hmac_keyring_file", "api_key_hmac_keyring_env", "delegation_hmac_keyring_file", "delegation_hmac_keyring_env", "reveal"}},
		{"global.services.access.credentials.reveal", nestedMapAt(global, "services", "access", "credentials", "reveal"), []string{"enabled", "kek_keyring_file", "kek_keyring_env"}},
		{"global.services.access.tenant_context", nestedMapAt(global, "services", "access", "tenant_context"), []string{"signing_key_file", "signing_key_env", "max_start_age"}},
		{"global.services.access.enforcement", nestedMapAt(global, "services", "access", "enforcement"), []string{"failure_mode", "request_accounting", "token_accounting", "unknown_usage_action", "settle_on", "deduplicate_by", "max_usage_backlog"}},
		{"global.services.access.usage_storage", nestedMapAt(global, "services", "access", "usage_storage"), []string{"create_ahead_months", "maintenance_interval", "raw_retention"}},
		{"global.services.backend_egress", nestedMapAt(global, "services", "backend_egress"), []string{"policy_file"}},
		{"global.services.backend_dispatch", nestedMapAt(global, "services", "backend_dispatch"), []string{"bind_address", "port", "audience", "capability_ttl", "max_request_body_bytes"}},
		{"global.services.management_api", nestedMapAt(global, "services", "management_api"), []string{"bind_address", "port", "remote_exposure", "auth", "tls"}},
		{"global.services.management_api.auth", nestedMapAt(global, "services", "management_api", "auth"), []string{"mode", "tokens", "roles", "token_signing_keyring_file", "token_signing_keyring_env", "service_account_hmac_keyring_file", "service_account_hmac_keyring_env", "invitation_hmac_keyring_file", "invitation_hmac_keyring_env", "control_plane_hmac_keyring_file", "control_plane_hmac_keyring_env", "response_kek_keyring_file", "response_kek_keyring_env", "bootstrap", "recovery"}},
		{"global.services.management_api.tls", nestedMapAt(global, "services", "management_api", "tls"), []string{"certificate_file", "certificate_env", "private_key_file", "private_key_env", "client_ca_bundle_file", "client_ca_bundle_env"}},
		{"global.services.management_api.auth.bootstrap", nestedMapAt(global, "services", "management_api", "auth", "bootstrap"), []string{"token_file", "token_env", "disable_after_first_cluster_admin"}},
		{"global.services.management_api.auth.recovery", nestedMapAt(global, "services", "management_api", "auth", "recovery"), []string{"enabled", "token_file", "token_env", "loopback_only"}},
	}
	for _, check := range checks {
		if err := rejectUnknownBootstrapFields(check.path, check.node, check.allowed); err != nil {
			return err
		}
	}
	for _, field := range []string{"rollout_groups", "required_rollout_groups"} {
		groups := nestedSlice(nestedMapAt(global, "control_plane", "provider_catalog")[field])
		for index, group := range groups {
			if err := rejectUnknownBootstrapFields(
				fmt.Sprintf("global.control_plane.provider_catalog.%s[%d]", field, index),
				nestedStringMap(group), []string{"plane", "id"},
			); err != nil {
				return err
			}
		}
	}
	backendCredentials := nestedMapAt(global, "services", "backend_credentials")
	credentialNames := make([]string, 0, len(backendCredentials))
	for name := range backendCredentials {
		if name == "provider_kek_keyring_file" || name == "provider_kek_keyring_env" {
			continue
		}
		credentialNames = append(credentialNames, name)
	}
	sort.Strings(credentialNames)
	for _, name := range credentialNames {
		if err := rejectUnknownBootstrapFields(
			"global.services.backend_credentials."+name,
			nestedStringMap(backendCredentials[name]),
			[]string{"credential_adapter_id", "secret_file", "secret_env"},
		); err != nil {
			return err
		}
	}
	return nil
}

func nestedSlice(value interface{}) []interface{} {
	values, _ := value.([]interface{})
	return values
}

func rejectUnknownBootstrapFields(path string, node map[string]interface{}, allowed []string) error {
	if len(node) == 0 {
		return nil
	}
	allowedSet := make(map[string]bool, len(allowed))
	for _, field := range allowed {
		allowedSet[field] = true
	}
	unknown := make([]string, 0)
	for field := range node {
		if !allowedSet[field] {
			unknown = append(unknown, field)
		}
	}
	if len(unknown) == 0 {
		return nil
	}
	sort.Strings(unknown)
	return fmt.Errorf("unsupported fields in %s: %s", path, strings.Join(unknown, ", "))
}
