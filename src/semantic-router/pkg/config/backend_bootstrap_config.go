package config

import (
	"fmt"
	"path/filepath"
	"regexp"
	"sort"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routingsnapshot"
)

var backendCredentialAdapterPattern = regexp.MustCompile(`^[a-z][a-z0-9._-]{0,127}$`)

// BackendCredentialsConfig separates provider-credential encryption from
// file-authored secret references. File credential names are inlined so a
// provider binding can reference a concise stable name.
type BackendCredentialsConfig struct {
	ProviderKEKKeyringFile string                             `yaml:"provider_kek_keyring_file,omitempty"`
	ProviderKEKKeyringEnv  string                             `yaml:"provider_kek_keyring_env,omitempty"`
	File                   map[string]BackendCredentialConfig `yaml:",inline"`
}

type BackendCredentialConfig struct {
	CredentialAdapterID string `yaml:"credential_adapter_id"`
	SecretFile          string `yaml:"secret_file,omitempty"`
	SecretEnv           string `yaml:"secret_env,omitempty"`
	// SecretValue exists only while compiling a public backend_refs[].api_key.
	// It is never serialized or returned through a configuration API.
	SecretValue string `yaml:"-" json:"-"`
}

func (c BackendCredentialConfig) String() string {
	return fmt.Sprintf(
		"BackendCredentialConfig{CredentialAdapterID:%q, SecretFile:%q, SecretEnv:%q, SecretValue:<redacted>}",
		c.CredentialAdapterID, c.SecretFile, c.SecretEnv,
	)
}

func (c BackendCredentialConfig) GoString() string { return c.String() }

type BackendEgressConfig struct {
	PolicyFile string `yaml:"policy_file,omitempty"`
}

func validateBackendBootstrap(
	durableCredentialAuthority bool,
	credentials BackendCredentialsConfig,
	egress BackendEgressConfig,
) error {
	providerKEKConfigured := credentials.ProviderKEKKeyringFile != "" || credentials.ProviderKEKKeyringEnv != ""
	if err := validateSecretSource(
		"global.services.backend_credentials.provider_kek_keyring",
		credentials.ProviderKEKKeyringFile,
		credentials.ProviderKEKKeyringEnv,
		durableCredentialAuthority,
	); err != nil {
		return err
	}
	if !durableCredentialAuthority && providerKEKConfigured {
		return fmt.Errorf(
			"global.services.backend_credentials.provider_kek_keyring requires global.stores.management.postgres",
		)
	}
	names := make([]string, 0, len(credentials.File))
	for name := range credentials.File {
		names = append(names, name)
	}
	sort.Strings(names)
	for _, name := range names {
		credential := credentials.File[name]
		if strings.TrimSpace(name) == "" || name != strings.TrimSpace(name) {
			return fmt.Errorf("global.services.backend_credentials contains an invalid empty credential name")
		}
		if err := validateBackendCredentialSource(
			"global.services.backend_credentials."+name+".secret", credential,
		); err != nil {
			return err
		}
		if !backendCredentialAdapterPattern.MatchString(credential.CredentialAdapterID) {
			return fmt.Errorf("global.services.backend_credentials.%s.credential_adapter_id is invalid", name)
		}
	}
	policyFile := strings.TrimSpace(egress.PolicyFile)
	if policyFile != egress.PolicyFile {
		return fmt.Errorf("global.services.backend_egress.policy_file must not contain surrounding whitespace")
	}
	if policyFile != "" && !filepath.IsAbs(policyFile) {
		return fmt.Errorf("global.services.backend_egress.policy_file must be an absolute path")
	}
	if policyFile == "" {
		return fmt.Errorf("global.services.backend_egress.policy_file is required")
	}
	return nil
}

func validateBackendCredentialRefs(
	durableCredentialAuthority bool,
	credentials BackendCredentialsConfig,
	snapshot *routingsnapshot.Snapshot,
) error {
	if snapshot == nil {
		return nil
	}
	// Durable publications already contain opaque provider credential IDs.
	// They are resolved by the durable credential registry at dispatch time and
	// must not be interpreted as file-authored bootstrap names here.
	if durableCredentialAuthority {
		return nil
	}
	for _, model := range snapshot.Models {
		for _, backend := range model.Backends {
			ref := strings.TrimSpace(backend.ProviderCredentialID)
			if ref != backend.ProviderCredentialID {
				return fmt.Errorf("model %q backend %q credential_ref must not contain surrounding whitespace", model.ID, backend.ID)
			}
			if ref == "" {
				continue
			}
			if _, ok := credentials.File[ref]; !ok {
				return fmt.Errorf("backend %q references undefined global.services.backend_credentials.%s", backend.ID, ref)
			}
		}
	}
	return nil
}

func cloneBackendCredentialsConfig(source BackendCredentialsConfig) BackendCredentialsConfig {
	cloned := source
	if source.File != nil {
		cloned.File = make(map[string]BackendCredentialConfig, len(source.File))
		for name, credential := range source.File {
			cloned.File[name] = credential
		}
	}
	return cloned
}

func validateBackendCredentialSource(path string, credential BackendCredentialConfig) error {
	configured := 0
	if credential.SecretFile != "" {
		configured++
	}
	if credential.SecretEnv != "" {
		configured++
	}
	if credential.SecretValue != "" {
		configured++
	}
	if configured != 1 {
		return fmt.Errorf("%s must configure exactly one secret source", path)
	}
	if credential.SecretValue != "" {
		if strings.TrimSpace(credential.SecretValue) != credential.SecretValue ||
			strings.ContainsAny(credential.SecretValue, "\r\n\x00") {
			return fmt.Errorf("%s literal value must not contain surrounding whitespace or control characters", path)
		}
		return nil
	}
	return validateSecretSource(path, credential.SecretFile, credential.SecretEnv, true)
}
