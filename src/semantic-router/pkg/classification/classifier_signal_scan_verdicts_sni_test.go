package classification

import (
	"crypto/tls"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/connector/connectortest"
)

func TestSharedGuardWindowsServerNameIdentity(t *testing.T) {
	for _, provider := range []string{config.SystemOneProvider, config.ModelRuntimeProvider} {
		t.Run(provider, func(t *testing.T) {
			_, store := sharedVerdictTestServer(t)
			var calls atomic.Int64
			handler := remoteVerdictTestHandler(&calls, func(r *http.Request) bool {
				return r.TLS.ServerName == "strict.example"
			})
			if provider == config.ModelRuntimeProvider {
				handler = attachedVerdictTestHandler(&calls)
			}
			ca := connectortest.NewPrivateCA(t, "*.example")
			upstream := httptest.NewUnstartedServer(handler)
			upstream.TLS = &tls.Config{MinVersion: tls.VersionTLS12, Certificates: []tls.Certificate{ca.Server}}
			upstream.StartTLS()
			t.Cleanup(upstream.Close)
			text := scanProse(4_000)
			for i, name := range []string{"lenient.example", "lenient.example", "strict.example"} {
				configure := func(d *config.ModelDeployment) {
					d.TLS = &config.DeploymentTLS{CAFile: ca.ChainFile, ServerName: name}
				}
				var classifier *Classifier
				if provider == config.ModelRuntimeProvider {
					classifier = attachedVerdictTestClassifier(t, store, upstream.URL, configure)
				} else {
					classifier = remoteVerdictTestClassifier(t, store, upstream.URL, "guard-v1", configure)
				}
				before := calls.Load()
				got := evaluateTurn(classifier, text)
				require.Empty(t, got.SignalErrors)
				require.Equal(t, name == "strict.example", guarded(got), "SNI must select its own policy")
				if i == 1 {
					require.Equal(t, before, calls.Load(), "unchanged server-only TLS should reuse completed windows")
				} else {
					require.Greater(t, calls.Load(), before, "a different TLS server name must scan")
				}
			}
		})
	}
}
