package main

import (
	"context"
	"crypto/tls"
	"io"
	"log"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/backendtelemetry"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/connector/connectortest"
)

const privateMetricsServerName = "llm.internal.example"

// An https backend's metrics are read with the backend's tls block: the
// private CA and server name that verify its chat calls, and the client
// certificate it requires.
func TestBackendTelemetryScrapesHTTPSBackendWithItsTLS(t *testing.T) {
	ca := connectortest.NewPrivateCA(t, privateMetricsServerName)
	engine := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.WriteString(w, "# TYPE vllm:num_requests_waiting gauge\nvllm:num_requests_waiting 3\n")
	}))
	engine.TLS = &tls.Config{
		Certificates: []tls.Certificate{ca.Server},
		ClientAuth:   tls.RequireAndVerifyClientCert,
		ClientCAs:    ca.ClientPool,
	}
	engine.Config.ErrorLog = log.New(io.Discard, "", 0)
	engine.StartTLS()
	defer engine.Close()
	port := engine.URL[strings.LastIndex(engine.URL, ":")+1:]

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
            server_name: ` + privateMetricsServerName + `
            cert_file: ` + ca.ClientCert + `
            key_file: ` + ca.ClientKey + `
global:
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
	defer func() { _ = part.Close(context.Background()) }()

	id := backendtelemetry.Identity{Model: "m", BackendRef: "private", Engine: backendtelemetry.EngineVLLM}
	deadline := time.Now().Add(5 * time.Second)
	for {
		if o, ok := part.store.Fresh(id); ok {
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
