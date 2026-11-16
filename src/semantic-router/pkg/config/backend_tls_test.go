package config

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func backendTLSDocument(defaults, backends string) string {
	return "version: v0.5\nproviders:\n  defaults:\n    model: private\n" + defaults + `  models:
    - name: private
      backend_refs:
` + backends + `recipes:
  - name: default
`
}

func TestBackendTLSReachesTheProviderProfile(t *testing.T) {
	cfg, err := ParseYAMLBytes([]byte(`version: v0.5
providers:
  defaults:
    model: private
    tls:
      ca_file: /etc/vllm-sr/certs/example-corp-ca.pem
  models:
    - name: private
      backend_refs:
        - name: internal
          provider: openai-compatible
          endpoint: https://llm-lb.internal.example/v1
          tls:
            server_name: llm.internal.example
            cert_file: /etc/vllm-sr/certs/router.crt
            key_file: /etc/vllm-sr/certs/router.key
    - name: partner
      backend_refs:
        - name: hosted
          provider: openai-compatible
          endpoint: https://partner.example.com/v1
          tls:
            ca_file: /etc/vllm-sr/certs/partner-ca.pem
    - name: public
      backend_refs:
        - name: hosted
          provider: openai
    - name: local
      backend_refs:
        - name: lab
          provider: vllm
          endpoint: vllm.internal.example:8000
recipes:
  - name: default
`))
	if err != nil {
		t.Fatal(err)
	}
	for endpoint, want := range map[string]*DeploymentTLS{
		"private_internal": {
			CAFile: "/etc/vllm-sr/certs/example-corp-ca.pem", ServerName: "llm.internal.example",
			CertFile: "/etc/vllm-sr/certs/router.crt", KeyFile: "/etc/vllm-sr/certs/router.key",
		},
		// A backend's own CA replaces the default.
		"partner_hosted": {CAFile: "/etc/vllm-sr/certs/partner-ca.pem"},
		// The default reaches an https backend without a tls block, here
		// through its provider's default base URL.
		"public_hosted": {CAFile: "/etc/vllm-sr/certs/example-corp-ca.pem"},
		// Plain HTTP has no TLS.
		"local_lab": nil,
	} {
		profile, err := cfg.GetProviderProfileForEndpoint(endpoint)
		if err != nil {
			t.Fatal(err)
		}
		switch {
		case want == nil && profile.TLS != nil:
			t.Errorf("%s: TLS = %+v, want none", endpoint, profile.TLS)
		case want != nil && (profile.TLS == nil || *profile.TLS != *want):
			t.Errorf("%s: TLS = %+v, want %+v", endpoint, profile.TLS, want)
		}
	}
}

func TestBackendTLSIsValidatedLikeADeployment(t *testing.T) {
	dir := t.TempDir()
	notPEM := filepath.Join(dir, "not-a-bundle.pem")
	if err := os.WriteFile(notPEM, []byte("not a certificate\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	const https = "        - provider: openai-compatible\n          endpoint: https://llm.internal.example/v1\n"
	for _, tc := range []struct {
		name     string
		defaults string
		backends string
		want     string
	}{
		{
			name:     "tls on plain HTTP",
			backends: "        - provider: vllm\n          endpoint: vllm.internal.example:8000\n          tls: {ca_file: /etc/ca.pem}\n",
			want:     "providers.models[private].backend_refs[0]: tls applies to an https:// endpoint",
		},
		{name: "relative CA file", backends: https + "          tls: {ca_file: certs/ca.pem}\n", want: "tls.ca_file must be an absolute path"},
		{name: "certificate without its key", backends: https + "          tls: {cert_file: /etc/router.crt}\n", want: "tls.cert_file and tls.key_file go together"},
		{name: "server name with a port", backends: https + "          tls: {server_name: \"llm.internal.example:443\"}\n", want: "tls.server_name must be a host name"},
		{
			name:     "skipped verification",
			backends: https + "          tls: {insecure_skip_verify: true}\n",
			want:     "tls.insecure_skip_verify applies to global.model_catalog.deployments; Envoy and the Router always verify a chat backend",
		},
		{name: "default without a CA file", defaults: "    tls: {}\n", backends: https, want: "providers.defaults.tls.ca_file must name a CA bundle"},
		{name: "relative default CA file", defaults: "    tls: {ca_file: ca.pem}\n", backends: https, want: "providers.defaults.tls.ca_file must be an absolute path"},
		{name: "default server name", defaults: "    tls: {ca_file: /etc/ca.pem, server_name: llm.internal.example}\n", backends: https, want: `unknown field "server_name"`},
		{name: "CA file that holds no certificate", backends: https + "          tls: {ca_file: " + notPEM + "}\n", want: "tls.ca_file " + notPEM + " holds no PEM certificate"},
		{name: "default CA file that holds no certificate", defaults: "    tls: {ca_file: " + notPEM + "}\n", backends: https, want: "providers.defaults.tls.ca_file " + notPEM + " holds no PEM certificate"},
		{
			name: "one pool, two CA files",
			backends: https + "          tls: {ca_file: /etc/ca-a.pem}\n          name: a\n" +
				https + "          tls: {ca_file: /etc/ca-b.pem}\n          name: b\n",
			want: "TLS files differ",
		},
		{
			name: "one pool, two server names",
			backends: https + "          tls: {server_name: a.internal.example}\n          name: a\n" +
				https + "          tls: {server_name: b.internal.example}\n          name: b\n",
			want: "TLS server name differ",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, err := ParseYAMLBytes([]byte(backendTLSDocument(tc.defaults, tc.backends)))
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("err = %v, want %q", err, tc.want)
			}
		})
	}
}

// A file that is not where the configuration is checked, as in the
// Dashboard's container, is read where the Router connects; a read-only
// check reads no file at all.
func TestBackendTLSFilesAreReadOnlyWhereTheyAre(t *testing.T) {
	missing := backendTLSDocument("    tls: {ca_file: /nonexistent/example-corp-ca.pem}\n",
		"        - provider: openai-compatible\n          endpoint: https://llm.internal.example/v1\n")
	if _, err := ParseYAMLBytes([]byte(missing)); err != nil {
		t.Fatalf("a CA file that is not here must not fail the load: %v", err)
	}

	notPEM := filepath.Join(t.TempDir(), "not-a-bundle.pem")
	if err := os.WriteFile(notPEM, []byte("not a certificate\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	present := backendTLSDocument("",
		"        - provider: openai-compatible\n          endpoint: https://llm.internal.example/v1\n          tls: {ca_file: "+notPEM+"}\n")
	if _, err := ParseYAMLBytesWithoutEnvExpansion([]byte(present)); err != nil {
		t.Fatalf("a read-only check must not read the CA file: %v", err)
	}
}

func TestBackendTLSSurvivesExport(t *testing.T) {
	cfg, err := ParseYAMLBytes([]byte(backendTLSDocument(
		"    tls: {ca_file: /etc/vllm-sr/certs/example-corp-ca.pem}\n",
		"        - provider: openai-compatible\n          endpoint: https://llm.internal.example/v1\n          tls: {server_name: llm.internal.example}\n",
	)))
	if err != nil {
		t.Fatal(err)
	}
	exported := CanonicalConfigFromRouterConfig(cfg)
	if got := exported.Providers.Defaults.TLS; got == nil || got.CAFile != "/etc/vllm-sr/certs/example-corp-ca.pem" {
		t.Fatalf("providers.defaults.tls = %+v", got)
	}
	backend := exported.Providers.Models[0].BackendRefs[0]
	if backend.TLS == nil || *backend.TLS != (DeploymentTLS{ServerName: "llm.internal.example"}) {
		t.Fatalf("backend_refs[0].tls = %+v, want the block as written", backend.TLS)
	}
}
