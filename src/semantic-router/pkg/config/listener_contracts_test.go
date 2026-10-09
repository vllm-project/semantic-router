package config

import (
	"path/filepath"
	"strings"
	"testing"
)

func TestListenerTLSNeedsACertificateAndAKey(t *testing.T) {
	for _, tls := range []*ListenerTLS{{CertFile: "tls.crt"}, {KeyFile: "tls.key"}, {CertFile: " ", KeyFile: "tls.key"}} {
		cfg := &RouterConfig{}
		cfg.Listeners = []Listener{{Name: "https", TLS: tls}}
		if err := validateListenerContracts(cfg); err == nil || !strings.Contains(err.Error(), "listener 'https'") {
			t.Fatalf("%+v: got %v", tls, err)
		}
	}
	cfg := &RouterConfig{}
	cfg.Listeners = []Listener{{Name: "https", TLS: &ListenerTLS{CertFile: "c", KeyFile: "k"}}, {Name: "http"}}
	if err := validateListenerContracts(cfg); err != nil {
		t.Fatal(err)
	}
}

func TestListenerTLSFilesResolveAgainstTheConfigDirectory(t *testing.T) {
	cert, key := ListenerTLS{CertFile: "certs/tls.crt", KeyFile: "/etc/tls/tls.key"}.Files("/app")
	if cert != filepath.Join("/app", "certs", "tls.crt") || key != "/etc/tls/tls.key" {
		t.Fatalf("got %s, %s", cert, key)
	}
}

func TestListenerTLSParsesFromTheCanonicalDocument(t *testing.T) {
	cfg, err := ParseYAMLBytes([]byte(canonicalTLSDocument))
	if err != nil {
		t.Fatal(err)
	}
	if tls := cfg.Listeners[0].TLS; tls == nil || tls.CertFile != "certs/tls.crt" || tls.KeyFile != "certs/tls.key" {
		t.Fatalf("tls = %+v", cfg.Listeners[0].TLS)
	}
	broken := strings.Replace(canonicalTLSDocument, "      key_file: certs/tls.key\n", "", 1)
	if _, err := ParseYAMLBytes([]byte(broken)); err == nil || !strings.Contains(err.Error(), "cert_file and key_file") {
		t.Fatalf("a listener with a certificate and no key must not load, got %v", err)
	}
}

const canonicalTLSDocument = `version: v0.3
listeners:
  - name: https
    address: 0.0.0.0
    port: 8443
    tls:
      cert_file: certs/tls.crt
      key_file: certs/tls.key
providers:
  defaults:
    model: m
  models:
    - name: m
      backend_refs:
        - endpoint: 127.0.0.1:8000
          protocol: http
          provider: vllm
routing:
  modelCards:
    - name: m
`

func TestSystemOneListenerScopeNeverInheritsChatModels(t *testing.T) {
	cfg := &RouterConfig{}
	cfg.ModelDeployments = map[string]ModelDeployment{
		"private-key": {Provider: ModelRuntimeProvider, Artifact: "vllm-sr/Vela-2.0-0.3B", PublicName: "judgment"},
	}
	listener := Listener{Name: "public", Models: []string{"chat-only"}}
	if err := validateListenerSystemOne(cfg, listener); err != nil {
		t.Fatal(err)
	}
	for _, names := range [][]string{{}, {"private-key"}, {"unknown"}, {" judgment "}, {"judgment", "judgment"}} {
		listener.SystemOne = &ListenerSystemOne{Models: names}
		if err := validateListenerSystemOne(cfg, listener); err == nil {
			t.Fatalf("accepted invalid public scope %v", names)
		}
	}
	listener.SystemOne = &ListenerSystemOne{Models: []string{"judgment"}}
	if err := validateListenerSystemOne(cfg, listener); err != nil {
		t.Fatal(err)
	}
	key, _, err := cfg.ResolveSystemOneDeployment("judgment")
	if err != nil || key != "private-key" {
		t.Fatalf("resolved %q: %v", key, err)
	}
	cfg.ModelDeployments["second-key"] = cfg.ModelDeployments["private-key"]
	if err := validateListenerSystemOne(cfg, listener); err == nil {
		t.Fatal("ambiguous public model accepted")
	}
}
