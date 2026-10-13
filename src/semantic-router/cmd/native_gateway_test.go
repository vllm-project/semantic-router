package main

import (
	"context"
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/tls"
	"crypto/x509"
	"encoding/pem"
	"errors"
	"math/big"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/configsnapshot"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/gateway"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routing"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/upstream"
)

func backendConfig(t *testing.T, url string) *config.RouterConfig {
	t.Helper()
	return backendConfigWithRouting(t, url, "")
}

// backendConfigWithRouting adds routing, indented under the routing key.
func backendConfigWithRouting(t *testing.T, url, routing string) *config.RouterConfig {
	t.Helper()
	cfg, err := config.ParseYAMLBytes([]byte(backendDocument(url, routing)))
	if err != nil {
		t.Fatal(err)
	}
	return cfg
}

func backendDocument(url, routing string) string {
	host := strings.TrimPrefix(url, "http://")
	return `version: v0.3
listeners:
  - name: http-8899
    address: 127.0.0.1
    port: 8899
providers:
  defaults:
    model: m
  models:
    - name: m
      backend_refs:
        - endpoint: ` + host + `
          protocol: http
          provider: vllm
global:
  stores:
    semantic_cache:
      enabled: false
routing:
  modelCards:
    - name: m
` + routing
}

type activateAll struct{}

func (activateAll) Validate(context.Context, *configsnapshot.Candidate) error { return nil }

func (activateAll) Warm(context.Context, *configsnapshot.Candidate) (configsnapshot.Warmed, error) {
	return activated{}, nil
}

type activated struct{}

func (activated) Activate(context.Context) error { return nil }
func (activated) Discard()                       {}

func upstreamOf(t *testing.T, snapshot *configsnapshot.Snapshot) *upstream.Set {
	t.Helper()
	set, ok := snapshot.Part(configsnapshot.ComponentUpstream).(*upstream.Set)
	if !ok {
		t.Fatal("the snapshot owns no upstream set")
	}
	return set
}

func callUpstream(set *upstream.Set) error {
	planned := &routing.Call{Route: "m", Request: routing.Request{Header: routing.Header{
		{Name: ":method", Value: http.MethodPost}, {Name: ":path", Value: "/v1/chat/completions"},
	}}}
	result, err := set.Execute(context.Background(), planned, "")
	if err != nil {
		return err
	}
	return result.Response.Body.Close()
}

func TestNativeUpstreamSetIsOwnedByTheSnapshot(t *testing.T) {
	var first, second atomic.Int32
	one := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) { first.Add(1) }))
	defer one.Close()
	two := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) { second.Add(1) }))
	defer two.Close()
	ctx := context.Background()
	manager := configsnapshot.NewManager(configsnapshot.Options{
		Runtime: activateAll{}, Parts: []configsnapshot.PartBuilder{upstreamPart(config.GatewayStandalone)},
	})
	apply := func(cfg *config.RouterConfig) (*configsnapshot.Snapshot, error) {
		return manager.Apply(ctx, configsnapshot.Update{Origin: configsnapshot.Origin{Source: configsnapshot.SourceKubernetes}, Config: cfg})
	}
	startup, err := manager.Install(ctx, configsnapshot.Update{
		Origin: configsnapshot.Origin{Source: configsnapshot.SourceStartup}, Config: backendConfig(t, one.URL),
	})
	if err != nil {
		t.Fatal(err)
	}
	moved, err := apply(backendConfig(t, two.URL))
	if err != nil {
		t.Fatal(err)
	}
	if upstreamOf(t, moved) == upstreamOf(t, startup) {
		t.Fatal("an endpoint change kept the old upstream set")
	}
	if err = callUpstream(upstreamOf(t, moved)); err != nil || second.Load() != 1 || first.Load() != 0 {
		t.Fatalf("calls = %d, %d (%v); the new snapshot must serve the new backend", first.Load(), second.Load(), err)
	}

	tuned, err := apply(backendConfigWithRouting(t, two.URL, `  signals:
    keywords:
      - name: urgent
        operator: OR
        keywords: ["urgent"]
`))
	if err != nil {
		t.Fatal(err)
	}
	if upstreamOf(t, tuned) != upstreamOf(t, moved) || !slices.Contains(tuned.Reused(), configsnapshot.ComponentUpstream) {
		t.Fatalf("a routing-only change rebuilt the upstream set (reused %v)", tuned.Reused())
	}

	if err = startup.Release(ctx); err != nil {
		t.Fatal(err)
	}
	if err = callUpstream(upstreamOf(t, startup)); err == nil {
		t.Fatal("the released snapshot's set still serves")
	}
	if err = moved.Release(ctx); err != nil {
		t.Fatal(err)
	}
	if err = callUpstream(upstreamOf(t, tuned)); err != nil {
		t.Fatalf("a set the active snapshot shares closed with its predecessor: %v", err)
	}
}

// TestExtProcModeStartsWithABackendTheRouterCannotCall pins the split: a
// backend Envoy accepts but the in-process client refuses (HTTPS to an IP
// literal) fails a standalone Router, while an ext_proc Router starts and only
// its request-graph hops find no route.
func TestExtProcModeStartsWithABackendTheRouterCannotCall(t *testing.T) {
	document := strings.Replace(backendDocument("http://192.0.2.10:8443", ""), "protocol: http", "protocol: https", 1)
	cfg, err := config.ParseYAMLBytes([]byte(document))
	if err != nil {
		t.Fatal(err)
	}
	install := func(mode config.GatewayMode) (*configsnapshot.Snapshot, error) {
		manager := configsnapshot.NewManager(configsnapshot.Options{
			Runtime: activateAll{}, Parts: []configsnapshot.PartBuilder{upstreamPart(mode)},
		})
		return manager.Install(context.Background(), configsnapshot.Update{
			Origin: configsnapshot.Origin{Source: configsnapshot.SourceStartup}, Config: cfg,
		})
	}
	if _, err = install(config.GatewayStandalone); err == nil || !strings.Contains(err.Error(), "DNS hostname") {
		t.Fatalf("standalone install = %v, want the backend refused", err)
	}
	snapshot, err := install(config.GatewayExtProc)
	if err != nil {
		t.Fatalf("ext_proc install: %v", err)
	}
	defer func() { _ = snapshot.Release(context.Background()) }()
	var refused *upstream.Error
	if err = callUpstream(upstreamOf(t, snapshot)); !errors.As(err, &refused) || refused.Kind != upstream.KindNoRoute {
		t.Fatalf("hop = %v, want no route", err)
	}
}

// TestActiveHealthChecksBelongToTheDataPathOwner pins who probes a model's
// backends: a standalone Router runs the active health checks the model
// configures, while an ext_proc Router leaves them to Envoy and its hops keep
// passive outlier detection only.
func TestActiveHealthChecksBelongToTheDataPathOwner(t *testing.T) {
	for _, mode := range []config.GatewayMode{config.GatewayStandalone, config.GatewayExtProc} {
		t.Run(string(mode), func(t *testing.T) {
			var probes, calls atomic.Int32
			handler := http.HandlerFunc(func(_ http.ResponseWriter, r *http.Request) {
				if r.URL.Path == "/health" {
					probes.Add(1)
					return
				}
				calls.Add(1)
			})
			one, two := httptest.NewServer(handler), httptest.NewServer(handler)
			defer one.Close()
			defer two.Close()
			document := strings.Replace(backendDocument(one.URL, ""), `      backend_refs:
        - endpoint: `+strings.TrimPrefix(one.URL, "http://"), `      reliability:
        consecutive_5xx: 3
        health_check_path: /health
        health_check_interval: 1h
      backend_refs:
        - endpoint: `+strings.TrimPrefix(two.URL, "http://")+`
          protocol: http
          provider: vllm
        - endpoint: `+strings.TrimPrefix(one.URL, "http://"), 1)
			cfg, err := config.ParseYAMLBytes([]byte(document))
			if err != nil {
				t.Fatal(err)
			}
			manager := configsnapshot.NewManager(configsnapshot.Options{
				Runtime: activateAll{}, Parts: []configsnapshot.PartBuilder{upstreamPart(mode)},
			})
			snapshot, err := manager.Install(context.Background(), configsnapshot.Update{
				Origin: configsnapshot.Origin{Source: configsnapshot.SourceStartup}, Config: cfg,
			})
			if err != nil {
				t.Fatal(err)
			}
			defer func() { _ = snapshot.Release(context.Background()) }()
			set := upstreamOf(t, snapshot)
			cluster := set.Topology().Clusters[0]
			if len(cluster.Endpoints) != 2 || cluster.Outlier == nil {
				t.Fatalf("cluster = %+v, want two endpoints under passive outlier detection", cluster)
			}
			if err = callUpstream(set); err != nil || calls.Load() != 1 {
				t.Fatalf("hop = %v after %d backend calls, want one answered call", err, calls.Load())
			}
			// The first checks run when the set warms, before the snapshot serves.
			standalone := mode == config.GatewayStandalone
			if got := probes.Load(); (cluster.HealthCheck != nil) != standalone || (got > 0) != standalone {
				t.Fatalf("health check %+v sent %d probes; want them only in standalone mode", cluster.HealthCheck, got)
			}
		})
	}
}

func TestNativeListenerBindingChangesNeedARestart(t *testing.T) {
	backend := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {}))
	defer backend.Close()
	ctx := context.Background()
	manager := configsnapshot.NewManager(configsnapshot.Options{
		Runtime: activateAll{}, Parts: []configsnapshot.PartBuilder{upstreamPart(config.GatewayStandalone)},
	})
	active, err := manager.Install(ctx, configsnapshot.Update{
		Origin: configsnapshot.Origin{Source: configsnapshot.SourceStartup}, Config: backendConfig(t, backend.URL),
	})
	if err != nil {
		t.Fatal(err)
	}
	for name, edit := range map[string]struct {
		change func(*config.RouterConfig)
		path   string
	}{
		"port":    {func(cfg *config.RouterConfig) { cfg.Listeners[0].Port = 8900 }, "listeners[http-8899].port"},
		"address": {func(cfg *config.RouterConfig) { cfg.Listeners[0].Address = "0.0.0.0" }, "listeners[http-8899].address"},
		"timeout": {func(cfg *config.RouterConfig) { cfg.Listeners[0].Timeout = "not-a-duration" }, "listeners[http-8899].timeout"},
		"added": {func(cfg *config.RouterConfig) {
			cfg.Listeners = append(cfg.Listeners, config.Listener{Name: "http-9000", Address: "127.0.0.1", Port: 9000})
		}, "listeners[http-9000]"},
		"removed": {func(cfg *config.RouterConfig) {
			cfg.Listeners[0].Name = "renamed"
		}, "listeners[renamed]"},
		"tls": {func(cfg *config.RouterConfig) {
			cfg.Listeners[0].TLS = &config.ListenerTLS{CertFile: "tls.crt", KeyFile: "tls.key"}
		}, "listeners[http-8899].tls"},
	} {
		t.Run(name, func(t *testing.T) {
			cfg := backendConfig(t, backend.URL)
			edit.change(cfg)
			_, applyErr := manager.Apply(ctx, configsnapshot.Update{Origin: configsnapshot.Origin{Source: configsnapshot.SourceKubernetes}, Config: cfg})
			reasons := configsnapshot.ReasonsOf(applyErr)
			if len(reasons) == 0 || reasons[0].Code != configsnapshot.CodeRestartRequired || reasons[0].Path != edit.path {
				t.Fatalf("reasons = %+v (%v), want restart_required at %s", reasons, applyErr, edit.path)
			}
			if manager.Active() != active {
				t.Fatal("a rejected listener change replaced the active snapshot")
			}
		})
	}
	keys := backendConfig(t, backend.URL)
	keys.Listeners[0].APIKeys = []string{"fake-client-key"}
	if _, err = manager.Apply(ctx, configsnapshot.Update{Origin: configsnapshot.Origin{Source: configsnapshot.SourceKubernetes}, Config: keys}); err != nil {
		t.Fatalf("client keys change without a restart: %v", err)
	}
}

func TestListenNativeBindsTheOverrideAddress(t *testing.T) {
	cfg := backendConfig(t, "http://127.0.0.1:1")
	cfg.Listeners[0].Address = "192.0.2.1"
	cfg.Listeners[0].Port = 0
	listeners, err := listenNative(cfg, "127.0.0.1", gateway.Options{Serving: gateway.Static(gateway.Serving{})})
	if err != nil {
		t.Fatalf("the override replaces a configured address the host cannot bind: %v", err)
	}
	defer func() {
		for _, l := range listeners {
			_ = l.listener.Close()
		}
	}()
	if host, _, _ := net.SplitHostPort(listeners[0].listener.Addr().String()); host != "127.0.0.1" {
		t.Fatalf("bound %s", listeners[0].listener.Addr())
	}
	if listeners[0].server.TLSConfig != nil {
		t.Fatal("a listener without tls serves cleartext")
	}
}

func TestListenerServerOptionsLoadTheListenersCertificate(t *testing.T) {
	cfg := backendConfig(t, "http://127.0.0.1:1")
	cfg.ConfigBaseDir = t.TempDir()
	certPEM, keyPEM := selfSignedPair(t)
	if err := os.WriteFile(filepath.Join(cfg.ConfigBaseDir, "tls.crt"), certPEM, 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(cfg.ConfigBaseDir, "tls.key"), keyPEM, 0o600); err != nil {
		t.Fatal(err)
	}
	spec := cfg.Listeners[0]
	spec.Timeout = "30s"
	spec.TLS = &config.ListenerTLS{CertFile: "tls.crt", KeyFile: "tls.key"}
	opts, err := listenerServerOptions(cfg, spec)
	if err != nil {
		t.Fatalf("relative paths resolve against the configuration's directory: %v", err)
	}
	if opts.TLS == nil || opts.TLS.GetCertificate == nil || opts.IdleTimeout != 30*time.Second {
		t.Fatalf("options = %+v", opts)
	}
	if cert, certErr := opts.TLS.GetCertificate(&tls.ClientHelloInfo{}); certErr != nil || cert == nil {
		t.Fatalf("the listener's certificate: %v", certErr)
	}
	spec.TLS = &config.ListenerTLS{CertFile: "missing.crt", KeyFile: "tls.key"}
	if _, err := listenerServerOptions(cfg, spec); err == nil || !strings.Contains(err.Error(), "tls:") {
		t.Fatalf("a certificate that cannot load must fail the listener, got %v", err)
	}
}

func selfSignedPair(t *testing.T) (certPEM, keyPEM []byte) {
	t.Helper()
	key, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		t.Fatal(err)
	}
	template := &x509.Certificate{SerialNumber: big.NewInt(1), NotBefore: time.Now(), NotAfter: time.Now().Add(time.Hour)}
	der, err := x509.CreateCertificate(rand.Reader, template, template, &key.PublicKey, key)
	if err != nil {
		t.Fatal(err)
	}
	keyDER, err := x509.MarshalECPrivateKey(key)
	if err != nil {
		t.Fatal(err)
	}
	return pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: der}),
		pem.EncodeToMemory(&pem.Block{Type: "EC PRIVATE KEY", Bytes: keyDER})
}

func TestListenerIdentityTrustComesFromTheNamedListener(t *testing.T) {
	cfg := &config.RouterConfig{APIServer: config.APIServer{Listeners: []config.Listener{
		{Name: "public", Port: 8899},
		{Name: "behind-auth", Port: 8900, Identity: &config.ListenerIdentity{
			TrustHeaders: true, TrustedPeers: []string{"10.0.0.0/8"},
		}},
	}}}
	if trust := listenerIdentityTrust(cfg, "public"); trust.Headers || len(trust.Peers) != 0 {
		t.Fatalf("a listener without identity trusts nothing: %+v", trust)
	}
	trust := listenerIdentityTrust(cfg, "behind-auth")
	if !trust.Headers || len(trust.Peers) != 1 || trust.Peers[0].String() != "10.0.0.0/8" {
		t.Fatalf("trust = %+v", trust)
	}
}
