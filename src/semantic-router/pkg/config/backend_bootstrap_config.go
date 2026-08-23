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

// BackendCredentialsConfig separates managed provider-credential encryption
// from standalone secret references. Standalone credential names are inlined
// so provider bindings can reference a concise stable name.
type BackendCredentialsConfig struct {
	ProviderKEKKeyringFile string                             `yaml:"provider_kek_keyring_file,omitempty"`
	ProviderKEKKeyringEnv  string                             `yaml:"provider_kek_keyring_env,omitempty"`
	Standalone             map[string]BackendCredentialConfig `yaml:",inline"`
}

type BackendCredentialConfig struct {
	CredentialAdapterID string `yaml:"credential_adapter_id"`
	SecretFile          string `yaml:"secret_file,omitempty"`
	SecretEnv           string `yaml:"secret_env,omitempty"`
}

type BackendEgressConfig struct {
	PolicyFile string `yaml:"policy_file,omitempty"`
}

func validateBackendBootstrap(mode string, credentials BackendCredentialsConfig, egress BackendEgressConfig) error {
	providerKEKConfigured := credentials.ProviderKEKKeyringFile != "" || credentials.ProviderKEKKeyringEnv != ""
	if err := validateSecretSource(
		"global.services.backend_credentials.provider_kek_keyring",
		credentials.ProviderKEKKeyringFile,
		credentials.ProviderKEKKeyringEnv,
		mode == ControlPlaneModeManaged,
	); err != nil {
		return err
	}
	if mode == ControlPlaneModeStandalone && providerKEKConfigured {
		return fmt.Errorf("global.services.backend_credentials.provider_kek_keyring is managed-only")
	}
	if mode == ControlPlaneModeManaged && len(credentials.Standalone) > 0 {
		return fmt.Errorf("managed mode rejects standalone backend credential definitions")
	}
	names := make([]string, 0, len(credentials.Standalone))
	for name := range credentials.Standalone {
		names = append(names, name)
	}
	sort.Strings(names)
	for _, name := range names {
		credential := credentials.Standalone[name]
		if strings.TrimSpace(name) == "" || name != strings.TrimSpace(name) {
			return fmt.Errorf("global.services.backend_credentials contains an invalid empty credential name")
		}
		if err := validateSecretSource(
			"global.services.backend_credentials."+name+".secret",
			credential.SecretFile,
			credential.SecretEnv,
			true,
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
		return fmt.Errorf("%s mode requires global.services.backend_egress.policy_file", mode)
	}
	return nil
}

func validateBackendCredentialRefs(
	mode string,
	credentials BackendCredentialsConfig,
	snapshot *routingsnapshot.Snapshot,
) error {
	if snapshot == nil {
		return nil
	}
	// Managed snapshots already contain opaque provider credential IDs issued
	// by the control plane. They are resolved by the managed credential
	// registry at dispatch time and must never be interpreted as standalone
	// bootstrap names here. Managed bootstrap cannot inline routing resources,
	// so there is no second path for a standalone credential_ref to enter.
	if mode == ControlPlaneModeManaged {
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
			if _, ok := credentials.Standalone[ref]; !ok {
				return fmt.Errorf("backend %q references undefined global.services.backend_credentials.%s", backend.ID, ref)
			}
		}
	}
	return nil
}

func cloneBackendCredentialsConfig(source BackendCredentialsConfig) BackendCredentialsConfig {
	cloned := source
	if source.Standalone != nil {
		cloned.Standalone = make(map[string]BackendCredentialConfig, len(source.Standalone))
		for name, credential := range source.Standalone {
			cloned.Standalone[name] = credential
		}
	}
	return cloned
}
