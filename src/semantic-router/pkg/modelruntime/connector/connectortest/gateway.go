// Package connectortest is a fake gateway that a deployment's connection
// block must get through: TLS with its own certificate authority, a path
// prefix, a key read from a file and sent in a custom header without a
// prefix, and a non-secret header.
package connectortest

import (
	"bytes"
	"encoding/pem"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

const (
	// Prefix is the path the gateway serves the API under.
	Prefix = "/Unified/v1"
	// Header carries the key, with no prefix.
	Header = "Ocp-Apim-Subscription-Key"
	// Key is the key the gateway accepts.
	Key = "gateway-test-key"
	// ServerName is the name the gateway's certificate is valid for.
	ServerName = "example.com"
)

// Call is one request the gateway answered.
type Call struct {
	Method, Path, Query string
	Body                []byte
}

// Gateway answers the operations it is given, under Prefix, only to a
// request that carries Key in Header and the X-Org header.
type Gateway struct {
	Server  *httptest.Server
	CAFile  string
	KeyFile string

	mu    sync.Mutex
	calls []Call
}

// New starts a gateway whose operations map a path below Prefix (such as
// "/classify") to its handler.
func New(t *testing.T, operations map[string]http.HandlerFunc) *Gateway {
	t.Helper()
	g := &Gateway{}
	g.Server = httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get(Header) != Key || r.Header.Get("Authorization") != "" {
			http.Error(w, "missing credential", http.StatusUnauthorized)
			return
		}
		if r.Header.Get("X-Org") != "acme" {
			http.Error(w, "missing header", http.StatusBadRequest)
			return
		}
		operation, ok := strings.CutPrefix(r.URL.Path, Prefix)
		handler := operations[operation]
		if !ok || handler == nil {
			http.Error(w, "unknown path "+r.URL.Path, http.StatusNotFound)
			return
		}
		body, err := io.ReadAll(r.Body)
		if err != nil {
			http.Error(w, "unreadable body", http.StatusBadRequest)
			return
		}
		g.mu.Lock()
		g.calls = append(g.calls, Call{Method: r.Method, Path: r.URL.Path, Query: r.URL.RawQuery, Body: body})
		g.mu.Unlock()
		r.Body = io.NopCloser(bytes.NewReader(body))
		handler(w, r)
	}))
	t.Cleanup(g.Server.Close)
	dir := t.TempDir()
	g.CAFile = filepath.Join(dir, "ca.pem")
	certificate := pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: g.Server.Certificate().Raw})
	if err := os.WriteFile(g.CAFile, certificate, 0o600); err != nil {
		t.Fatal(err)
	}
	g.KeyFile = filepath.Join(dir, "key")
	if err := os.WriteFile(g.KeyFile, []byte(Key+"\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	return g
}

// Deployment is a deployment of provider that reaches the gateway through
// its connection block, with model as its remote model.
func (g *Gateway) Deployment(provider, model string) config.ModelDeployment {
	empty := ""
	return config.ModelDeployment{
		Provider: provider,
		Endpoint: g.Server.URL + Prefix,
		Model:    model,
		Auth:     &config.DeploymentAuth{Header: Header, Prefix: &empty, APIKeyFile: g.KeyFile},
		Headers:  map[string]string{"X-Org": "acme"},
		TLS:      &config.DeploymentTLS{CAFile: g.CAFile, ServerName: ServerName},
	}
}

// Calls lists the requests the gateway answered.
func (g *Gateway) Calls() []Call {
	g.mu.Lock()
	defer g.mu.Unlock()
	return append([]Call(nil), g.calls...)
}
