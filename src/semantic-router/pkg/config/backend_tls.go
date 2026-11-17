package config

import (
	"errors"
	"fmt"
	"io/fs"
	"net/url"
	"os"
	"path/filepath"
	"strings"

	modelcatalog "github.com/vllm-project/semantic-router/src/semantic-router/pkg/catalog"
)

// ProviderDefaultsTLS is the CA bundle of every https backend whose own tls
// names none. Like a backend's ca_file it replaces the system roots, so it
// must hold every CA those backends chain to.
type ProviderDefaultsTLS struct {
	CAFile string `yaml:"ca_file,omitempty" json:"ca_file,omitempty"`
}

func (t *ProviderDefaultsTLS) validate() error {
	if t == nil {
		return nil
	}
	if strings.TrimSpace(t.CAFile) == "" {
		return fmt.Errorf("tls.ca_file must name a CA bundle")
	}
	if !filepath.IsAbs(t.CAFile) {
		return fmt.Errorf("tls.ca_file must be an absolute path")
	}
	return nil
}

func cloneProviderDefaultsTLS(value *ProviderDefaultsTLS) *ProviderDefaultsTLS {
	if value == nil {
		return nil
	}
	clone := *value
	return &clone
}

func cloneDeploymentTLS(value *DeploymentTLS) *DeploymentTLS {
	if value == nil {
		return nil
	}
	clone := *value
	return &clone
}

// validateTLS checks a backend's tls block as a deployment's, except that a
// chat backend is always verified.
func (b CanonicalBackendRef) validateTLS(builtIn *modelcatalog.Registry) error {
	if b.TLS == nil {
		return nil
	}
	if b.TLS.InsecureSkipVerify {
		return fmt.Errorf("tls.insecure_skip_verify applies to global.model_catalog.deployments; Envoy and the Router always verify a chat backend, so name its CA in tls.ca_file")
	}
	if !isHTTPSEndpoint(backendEndpointURL(builtIn, b)) {
		return fmt.Errorf("tls applies to an https:// endpoint")
	}
	return b.TLS.validateShape()
}

// backendEndpointURL is the URL a backend is reached at: its endpoint, or
// its provider's default base URL when it names none.
func backendEndpointURL(builtIn *modelcatalog.Registry, backend CanonicalBackendRef) string {
	if endpoint := canonicalBackendURL(backend); endpoint != "" {
		return endpoint
	}
	providerID := strings.TrimSpace(backend.Provider)
	if providerID == "" {
		providerID = "vllm"
	}
	if definition, ok := builtIn.Provider(providerID); ok {
		return definition.DefaultBaseURL
	}
	return ""
}

func isHTTPSEndpoint(endpoint string) bool {
	parsed, err := url.Parse(endpointURLForParsing(endpoint))
	return err == nil && parsed.Scheme == "https"
}

// effectiveBackendTLS is the tls an https backend connects with: its own
// block, with the default CA when the block names none. A plain-HTTP
// backend has none.
func effectiveBackendTLS(backend *DeploymentTLS, defaults *ProviderDefaultsTLS, endpoint string) *modelcatalog.ProviderTLS {
	if !isHTTPSEndpoint(endpoint) {
		return nil
	}
	var effective modelcatalog.ProviderTLS
	if backend != nil {
		effective = modelcatalog.ProviderTLS{
			CAFile: backend.CAFile, ServerName: backend.ServerName,
			CertFile: backend.CertFile, KeyFile: backend.KeyFile,
		}
	}
	if effective.CAFile == "" && defaults != nil {
		effective.CAFile = defaults.CAFile
	}
	if effective == (modelcatalog.ProviderTLS{}) {
		return nil
	}
	return &effective
}

func deploymentTLSFromCatalog(value *modelcatalog.ProviderTLS) *DeploymentTLS {
	if value == nil {
		return nil
	}
	return &DeploymentTLS{
		CAFile: value.CAFile, ServerName: value.ServerName,
		CertFile: value.CertFile, KeyFile: value.KeyFile,
	}
}

// validateBackendTLSFiles reads the files the backends' tls blocks name, as
// the Router and Envoy read them to connect. A file that is not there is
// skipped: the configuration may be checked where it is not mounted, such as
// in the Dashboard, and whoever connects reads it again.
func validateBackendTLSFiles(cfg *RouterConfig) error {
	if cfg.SkipExternalAssetValidation {
		return nil
	}
	if defaults := cfg.DefaultBackendTLS; defaults != nil {
		if err := checkPresentTLSFiles(&DeploymentTLS{CAFile: defaults.CAFile}); err != nil {
			return fmt.Errorf("providers.defaults.%w", err)
		}
	}
	for _, alias := range cfg.ProviderModelOrder {
		authored := cfg.ModelConfig[alias].AuthoredModel
		if authored == nil {
			continue
		}
		for index, backend := range authored.BackendRefs {
			if err := checkPresentTLSFiles(backend.TLS); err != nil {
				return fmt.Errorf("providers.models[%s].backend_refs[%d]: %w", alias, index, err)
			}
		}
	}
	return nil
}

func checkPresentTLSFiles(t *DeploymentTLS) error {
	if t == nil {
		return nil
	}
	present := DeploymentTLS{}
	if fileIsThere(t.CAFile) {
		present.CAFile = t.CAFile
	}
	if fileIsThere(t.CertFile) && fileIsThere(t.KeyFile) {
		present.CertFile, present.KeyFile = t.CertFile, t.KeyFile
	}
	_, err := present.ClientConfig()
	return err
}

func fileIsThere(path string) bool {
	if path == "" {
		return false
	}
	_, err := os.Stat(path)
	return !errors.Is(err, fs.ErrNotExist)
}
