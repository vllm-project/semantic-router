package extproc

import (
	"context"
	"crypto/tls"
	"crypto/x509"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/http/httptest"
	"net/url"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"
	"go.uber.org/zap/zaptest/observer"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/connector/connectortest"
)

const privateBackendName = "llm.internal.example"

// privateBackend serves handler over TLS with a certificate an intermediate
// issued, and sends only that certificate, as a server that omits its chain
// does. It returns the PKI and the port.
func privateBackend(t *testing.T, handler http.Handler) (*connectortest.PrivateCA, string) {
	t.Helper()
	ca := connectortest.NewPrivateCA(t, privateBackendName)
	server := httptest.NewUnstartedServer(handler)
	server.TLS = &tls.Config{Certificates: []tls.Certificate{ca.Server}}
	server.Config.ErrorLog = log.New(io.Discard, "", 0)
	server.StartTLS()
	t.Cleanup(server.Close)
	parsed, err := url.Parse(server.URL)
	require.NoError(t, err)
	return ca, parsed.Port()
}

// backendTrustCases are the trust bundles a backend's tls block can name:
// none (the system roots), the root alone, and the intermediate with the
// root. Only the last verifies a server that does not send its intermediate.
func backendTrustCases(ca *connectortest.PrivateCA) []struct {
	name    string
	tls     string
	trusted bool
} {
	return []struct {
		name    string
		tls     string
		trusted bool
	}{
		{name: "system roots", tls: "{server_name: " + privateBackendName + "}"},
		{name: "root without the intermediate", tls: fmt.Sprintf("{ca_file: %s, server_name: %s}", ca.RootFile, privateBackendName)},
		{name: "intermediate and root", tls: fmt.Sprintf("{ca_file: %s, server_name: %s}", ca.ChainFile, privateBackendName), trusted: true},
	}
}

func TestFallbackHTTPDispatchTrustsTheBackendCA(t *testing.T) {
	ca, port := privateBackend(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"id":"chatcmpl-tls","object":"chat.completion","created":1,"model":"private-llm",`+
			`"choices":[{"index":0,"message":{"role":"assistant","content":"verified"},"finish_reason":"stop"}]}`)
	}))
	for _, tc := range backendTrustCases(ca) {
		t.Run(tc.name, func(t *testing.T) {
			cfg, err := config.ParseYAMLBytes([]byte(fmt.Sprintf(`version: v0.5
providers:
  defaults:
    model: private-llm
  models:
    - name: private-llm
      backend_refs:
        - name: internal
          provider: openai-compatible
          endpoint: https://localhost:%s/v1
          tls: %s
recipes:
  - name: default
`, port, tc.tls)))
			require.NoError(t, err)
			address, backendName, found, err := cfg.ResolvePrimaryBackendForModel("private-llm")
			require.NoError(t, err)
			require.True(t, found)
			profile, err := cfg.GetProviderProfileForEndpoint(backendName)
			require.NoError(t, err)
			router := &OpenAIRouter{Config: cfg, CredentialResolver: newTestCredentialResolver(cfg)}
			dispatch := &providerDispatch{
				logicalModel: "private-llm", backendName: backendName, backendAddress: address,
				profile: profile, targetFormat: llmprotocol.OpenAIChatV1,
			}

			body, status, err := router.dispatchFallbackHTTP(context.Background(), &RequestContext{}, dispatch,
				[]byte(`{"model":"private-llm","messages":[{"role":"user","content":"hi"}]}`), nil)

			if !tc.trusted {
				var unknownAuthority x509.UnknownAuthorityError
				require.ErrorAsf(t, err, &unknownAuthority, "the fallback attempt must fail verification, got status %d", status)
				return
			}
			require.NoError(t, err)
			assert.Equal(t, http.StatusOK, status)
			assert.Contains(t, string(body), "verified")
		})
	}
}

// The probe that reads a backend's served context window trusts the same CA.
// It retries a failed handshake as it retries a backend still loading, so an
// untrusted backend is probed until the generation retires, here a deadline.
func TestServedContextWindowProbeTrustsTheBackendCA(t *testing.T) {
	interval := servedContextWindowRetryInterval
	servedContextWindowRetryInterval = 10 * time.Millisecond
	t.Cleanup(func() { servedContextWindowRetryInterval = interval })
	var calls atomic.Int32
	ca, port := privateBackend(t, vllmModelsHandler(t, &calls, nil, "Qwen/Qwen3.6-27B", intPtr(32_768)))
	for _, tc := range backendTrustCases(ca) {
		t.Run(tc.name, func(t *testing.T) {
			core, logs := observer.New(zapcore.DebugLevel)
			t.Cleanup(zap.ReplaceGlobals(zap.New(core)))
			cfg, err := config.ParseYAMLBytes([]byte(fmt.Sprintf(`version: v0.5
listeners: []
providers:
  defaults:
    model: qwen-long
  models:
    - name: qwen-long
      catalog: qwen/qwen3.6-27b
      backend_refs:
        - name: qwen-vllm
          provider: vllm
          endpoint: localhost:%s
          protocol: https
          api_key: backend-secret
          tls: %s
recipes:
  - name: default
global: {}
`, port, tc.tls)))
			require.NoError(t, err)
			targets := servedContextWindowTargets(cfg)
			require.Len(t, targets, 1)
			ctx, cancel := context.WithTimeout(context.Background(), time.Second)
			defer cancel()
			before := calls.Load()

			(&OpenAIRouter{Config: cfg}).checkServedContextWindow(ctx, targets[0])

			windows := logs.FilterMessage(servedWindowEvent).All()
			if !tc.trusted {
				assert.Equal(t, before, calls.Load(), "an unverified backend never receives the probe")
				assert.Empty(t, windows)
				return
			}
			assert.Equal(t, before+1, calls.Load())
			require.Len(t, windows, 1, "the probe must read the served window")
		})
	}
}

// A shadow copy uses the backend's configured certificate trust.
func TestShadowConnectorTrustsTheBackendCA(t *testing.T) {
	backendTLS := &tls.Config{ServerName: privateBackendName, RootCAs: x509.NewCertPool()}
	options := shadowConnectorOptions(&shadowJob{timeout: time.Second, responseLimit: 1024}, backendTLS)
	require.Same(t, backendTLS, options.TLSConfig)
	assert.False(t, options.TLSConfig.InsecureSkipVerify)
}

func TestProviderTLSConfigLoadsTheBackendTrust(t *testing.T) {
	ca := connectortest.NewPrivateCA(t, privateBackendName)
	tlsConfig, err := providerTLSConfig(&config.ProviderProfile{TLS: &config.DeploymentTLS{
		CAFile: ca.ChainFile, ServerName: privateBackendName, CertFile: ca.ClientCert, KeyFile: ca.ClientKey,
	}})
	require.NoError(t, err)
	assert.Equal(t, privateBackendName, tlsConfig.ServerName)
	assert.NotNil(t, tlsConfig.RootCAs)
	assert.Len(t, tlsConfig.Certificates, 1)

	tlsConfig, err = providerTLSConfig(&config.ProviderProfile{})
	require.NoError(t, err)
	assert.Nil(t, tlsConfig, "a backend without tls keeps the system roots")

	_, err = providerTLSConfig(&config.ProviderProfile{TLS: &config.DeploymentTLS{CAFile: ca.ClientKey}})
	require.ErrorContains(t, err, "holds no PEM certificate")
}
