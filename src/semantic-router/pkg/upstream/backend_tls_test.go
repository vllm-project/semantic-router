package upstream

import (
	"crypto/tls"
	"crypto/x509"
	"io"
	"log"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/connector/connectortest"
)

const privateServerName = "llm.internal.example"

// privateTLSBackend serves over TLS with a certificate an intermediate
// issued, sending only that certificate. requireClient also demands a
// client certificate the intermediate issued.
func privateTLSBackend(t *testing.T, requireClient bool) (*httptest.Server, *connectortest.PrivateCA) {
	t.Helper()
	ca := connectortest.NewPrivateCA(t, privateServerName)
	server := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.WriteString(w, "verified")
	}))
	server.TLS = &tls.Config{Certificates: []tls.Certificate{ca.Server}}
	if requireClient {
		server.TLS.ClientAuth = tls.RequireAndVerifyClientCert
		server.TLS.ClientCAs = ca.ClientPool
	}
	server.Config.ErrorLog = log.New(io.Discard, "", 0)
	server.StartTLS()
	t.Cleanup(server.Close)
	return server, ca
}

func privateCluster(name string, spec *TLSSpec) ClusterSpec {
	cluster := clusterOf(name, EndpointSpec{Name: "internal", Scheme: "https", Host: privateServerName, Port: 443, Weight: 1})
	cluster.TLS = spec
	return cluster
}

func TestCompileTakesTheBackendTLS(t *testing.T) {
	topology := compileYAML(t, `version: v0.5
providers:
  defaults:
    model: private
    tls:
      ca_file: /etc/vllm-sr/certs/example-corp-ca.pem
  models:
    - name: private
      backend_refs:
        - provider: openai-compatible
          endpoint: https://llm-lb.internal.example/v1
          tls:
            server_name: llm.internal.example
            cert_file: /etc/vllm-sr/certs/router.crt
            key_file: /etc/vllm-sr/certs/router.key
    - name: partner
      backend_refs:
        - provider: openai-compatible
          endpoint: https://partner.example.com/v1
          tls:
            ca_file: /etc/vllm-sr/certs/partner-ca.pem
    - name: local
      backend_refs:
        - provider: vllm
          endpoint: vllm.internal.example:8000
recipes:
  - name: default
`)
	want := map[string]*TLSSpec{
		"private": {
			ServerName: privateServerName, CAFile: "/etc/vllm-sr/certs/example-corp-ca.pem",
			CertFile: "/etc/vllm-sr/certs/router.crt", KeyFile: "/etc/vllm-sr/certs/router.key",
		},
		"partner": {ServerName: "partner.example.com", CAFile: "/etc/vllm-sr/certs/partner-ca.pem"},
		"local":   nil,
	}
	for name, spec := range want {
		cluster := topology.cluster(name)
		if cluster == nil {
			t.Fatalf("no cluster %s", name)
		}
		switch {
		case spec == nil && cluster.TLS != nil:
			t.Errorf("%s: TLS = %+v, want plain HTTP", name, cluster.TLS)
		case spec != nil && (cluster.TLS == nil || *cluster.TLS != *spec):
			t.Errorf("%s: TLS = %+v, want %+v", name, cluster.TLS, spec)
		}
	}
}

func TestDoTrustsTheClusterCAFile(t *testing.T) {
	server, ca := privateTLSBackend(t, false)
	for _, tc := range []struct {
		name    string
		caFile  string
		trusted bool
	}{
		{name: "system roots"},
		{name: "root without the intermediate", caFile: ca.RootFile},
		{name: "intermediate and root", caFile: ca.ChainFile, trusted: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			set := newSet(t, Options{Dial: dialTo(server.Listener.Addr().String())},
				privateCluster("private", &TLSSpec{ServerName: privateServerName, CAFile: tc.caFile}))

			resp, err := set.Do(t.Context(), post("private"))
			if err != nil {
				t.Fatal(err)
			}
			body, _ := readAll(t, resp)
			if tc.trusted {
				if resp.Local != nil || body != "verified" {
					t.Fatalf("body = %q, failure = %v; want the backend's answer", body, resp.Local)
				}
				return
			}
			var unknownAuthority x509.UnknownAuthorityError
			if KindOf(resp.Local) != KindConnectFailure || !errorsAs(error(resp.Local), &unknownAuthority) {
				t.Fatalf("failure = %v, want an unknown-authority connect failure", resp.Local)
			}
		})
	}
}

func TestDoPresentsTheClusterClientCertificate(t *testing.T) {
	server, ca := privateTLSBackend(t, true)
	withCertificate := &TLSSpec{ServerName: privateServerName, CAFile: ca.ChainFile, CertFile: ca.ClientCert, KeyFile: ca.ClientKey}
	set := newSet(t, Options{Dial: dialTo(server.Listener.Addr().String())},
		privateCluster("mtls", withCertificate),
		privateCluster("anonymous", &TLSSpec{ServerName: privateServerName, CAFile: ca.ChainFile}))

	resp, err := set.Do(t.Context(), post("mtls"))
	if err != nil {
		t.Fatal(err)
	}
	if body, _ := readAll(t, resp); resp.Local != nil || body != "verified" {
		t.Fatalf("body = %q, failure = %v; want the backend's answer", body, resp.Local)
	}

	resp, err = set.Do(t.Context(), post("anonymous"))
	if err != nil {
		t.Fatal(err)
	}
	if body, _ := readAll(t, resp); resp.Local == nil {
		t.Fatalf("body = %q; the backend must refuse a client without a certificate", body)
	}
}

// Clusters share a connection pool only when they trust the same files.
func TestClustersShareAPoolOnlyWithTheSameTLSFiles(t *testing.T) {
	_, ca := privateTLSBackend(t, false)
	set := newSet(t, Options{},
		privateCluster("chain", &TLSSpec{ServerName: privateServerName, CAFile: ca.ChainFile}),
		privateCluster("chain-again", &TLSSpec{ServerName: privateServerName, CAFile: ca.ChainFile}),
		privateCluster("root", &TLSSpec{ServerName: privateServerName, CAFile: ca.RootFile}))

	if set.clusters["chain"].pool != set.clusters["chain-again"].pool {
		t.Fatal("clusters with the same TLS files must share a pool")
	}
	if set.clusters["chain"].pool == set.clusters["root"].pool {
		t.Fatal("clusters with different CA files must not share a pool")
	}
}

func TestNewRefusesAClusterWhoseTLSFilesCannotBeLoaded(t *testing.T) {
	_, ca := privateTLSBackend(t, false)
	for _, tc := range []struct {
		name string
		spec TLSSpec
		want string
	}{
		{name: "missing CA file", spec: TLSSpec{CAFile: "/nonexistent/example-corp-ca.pem"}, want: "tls.ca_file /nonexistent/example-corp-ca.pem cannot be read"},
		{name: "CA file without a certificate", spec: TLSSpec{CAFile: ca.ClientKey}, want: "holds no PEM certificate"},
		{name: "certificate without its key", spec: TLSSpec{CertFile: ca.ClientCert, KeyFile: ca.RootFile}, want: "tls.cert_file and tls.key_file cannot be loaded"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			spec := tc.spec
			spec.ServerName = privateServerName
			_, err := New(Topology{Clusters: []ClusterSpec{privateCluster("private", &spec)}}, Options{})
			if err == nil || !strings.Contains(err.Error(), tc.want) || !strings.Contains(err.Error(), "cluster private") {
				t.Fatalf("err = %v, want it to name the cluster and %q", err, tc.want)
			}
		})
	}
}
