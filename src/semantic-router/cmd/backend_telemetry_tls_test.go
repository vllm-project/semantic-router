package main

import (
	"context"
	"crypto/tls"
	"io"
	"log"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/backendtelemetry"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/connector/connectortest"
)

const privateMetricsServerName = "llm.internal.example"

// mtlsEngine serves handler over TLS with the private CA's server
// certificate and requires a client certificate that CA issued.
func mtlsEngine(t *testing.T, ca *connectortest.PrivateCA, handler http.HandlerFunc) *httptest.Server {
	t.Helper()
	engine := httptest.NewUnstartedServer(handler)
	engine.TLS = &tls.Config{
		Certificates: []tls.Certificate{ca.Server},
		ClientAuth:   tls.RequireAndVerifyClientCert,
		ClientCAs:    ca.ClientPool,
	}
	engine.Config.ErrorLog = log.New(io.Discard, "", 0)
	engine.StartTLS()
	t.Cleanup(engine.Close)
	return engine
}

func serverPort(server *httptest.Server) string {
	return server.URL[strings.LastIndex(server.URL, ":")+1:]
}

// startMTLSTelemetry collects from one https backend on localhost whose tls
// block names the private CA, the client certificate and tlsExtra.
func startMTLSTelemetry(t *testing.T, ca *connectortest.PrivateCA, port, tlsExtra string) *backendTelemetryRuntime {
	t.Helper()
	cfg, err := config.ParseYAMLBytes([]byte(`version: v0.5
providers:
  defaults:
    model: m
  models:
    - name: m
      backend_refs:
        - name: private
          endpoint: localhost:` + port + `
          protocol: https
          provider: vllm
          engine: vllm
          tls:
            ca_file: ` + ca.ChainFile + `
            cert_file: ` + ca.ClientCert + `
            key_file: ` + ca.ClientKey + `
` + tlsExtra + `global:
  services:
    observability:
      backend_telemetry:
        enabled: true
        poll_interval: 20ms
`))
	if err != nil {
		t.Fatal(err)
	}
	part, err := startBackendTelemetry(cfg, backendTelemetryRegistry())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = part.Close(context.Background()) })
	return part
}

var privateBackend = backendtelemetry.Identity{Model: "m", BackendRef: "private", Engine: backendtelemetry.EngineVLLM}

// An https backend's metrics are read with the backend's tls block: the
// private CA and server name that verify its chat calls, and the client
// certificate it requires.
func TestBackendTelemetryScrapesHTTPSBackendWithItsTLS(t *testing.T) {
	ca := connectortest.NewPrivateCA(t, privateMetricsServerName)
	engine := mtlsEngine(t, ca, func(w http.ResponseWriter, _ *http.Request) {
		_, _ = io.WriteString(w, "# TYPE vllm:num_requests_waiting gauge\nvllm:num_requests_waiting 3\n")
	})
	part := startMTLSTelemetry(t, ca, serverPort(engine), "            server_name: "+privateMetricsServerName+"\n")

	deadline := time.Now().Add(5 * time.Second)
	for {
		if o, ok := part.store.Fresh(privateBackend); ok {
			if depth, _ := o.QueueDepth.Get(); depth != 3 {
				t.Fatalf("queue depth = %d, want 3", depth)
			}
			return
		}
		if time.Now().After(deadline) {
			t.Fatal("the https backend was never scraped with its tls block")
		}
		time.Sleep(5 * time.Millisecond)
	}
}

// A backend that redirects its metrics to another host the same CA vouches
// for must not receive the backend's client certificate there, nor have that
// host's load recorded as its own.
func TestBackendTelemetryRefusesARedirectToAnotherHost(t *testing.T) {
	ca := connectortest.NewPrivateCA(t, "localhost")
	var otherRequests atomic.Int64
	other := mtlsEngine(t, ca, func(w http.ResponseWriter, _ *http.Request) {
		otherRequests.Add(1)
		_, _ = io.WriteString(w, "# TYPE vllm:num_requests_waiting gauge\nvllm:num_requests_waiting 9\n")
	})
	var redirects atomic.Int64
	backend := mtlsEngine(t, ca, func(w http.ResponseWriter, r *http.Request) {
		redirects.Add(1)
		http.Redirect(w, r, "https://localhost:"+serverPort(other)+"/metrics", http.StatusFound)
	})
	part := startMTLSTelemetry(t, ca, serverPort(backend), "")

	deadline := time.Now().Add(5 * time.Second)
	for redirects.Load() < 3 {
		if time.Now().After(deadline) {
			t.Fatal("the backend was never scraped")
		}
		time.Sleep(5 * time.Millisecond)
	}
	if n := otherRequests.Load(); n != 0 {
		t.Fatalf("the redirect target received %d requests carrying the backend's client certificate", n)
	}
	if o, ok := part.store.Fresh(privateBackend); ok {
		t.Fatalf("a redirected scrape recorded %+v for the backend", o)
	}
}
