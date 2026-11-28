package classification

import (
	"bytes"
	"crypto/tls"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"sync/atomic"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/connector/connectortest"
)

func TestSharedGuardWindowsClientCertificateChange(t *testing.T) {
	for _, shared := range []bool{false, true} {
		name := "local"
		if shared {
			name = "shared"
		}
		t.Run(name, func(t *testing.T) {
			server, store := sharedVerdictTestServer(t)
			if !shared {
				store = nil
			}
			var calls atomic.Int64
			upstream, lenient, strict := remoteVerdictMTLSTestServer(t, &calls)
			// Replicas can mount different certificates at the same paths.
			// Rotating their contents must not preserve another policy's verdict.
			connection := lenient
			dir := t.TempDir()
			connection.CertFile = filepath.Join(dir, "client.crt")
			connection.KeyFile = filepath.Join(dir, "client.key")
			text := scanProse(4_000)
			var classifiers []*Classifier
			for i, identity := range []config.DeploymentTLS{lenient, strict, lenient} {
				for destination, source := range map[string]string{
					connection.CertFile: identity.CertFile,
					connection.KeyFile:  identity.KeyFile,
				} {
					data, err := os.ReadFile(source)
					require.NoError(t, err)
					require.NoError(t, os.WriteFile(destination, data, 0o600))
				}
				classifier := remoteVerdictTestClassifier(t, store, upstream.URL, "guard-v1", func(deployment *config.ModelDeployment) {
					deployment.TLS = &connection
				})
				classifiers = append(classifiers, classifier)
				before := calls.Load()
				got := evaluateTurn(classifier, text)
				require.Empty(t, got.SignalErrors)
				require.Equal(t, i == 1, guarded(got), "a client certificate must select its own guard policy")
				require.Greater(t, calls.Load(), before, "each replica must ask the guard with its own certificate")
			}
			for _, classifier := range classifiers {
				require.Empty(t, classifier.windowVerdicts.entries, "client-authenticated deployments must not retain window verdicts")
			}
			require.Empty(t, server.Keys(), "client-authenticated deployments must not publish window verdicts")
			require.Zero(t, server.CommandCount(), "client-authenticated deployments must not read or write Redis")
		})
	}
}

func TestSharedGuardWindowsClientCertificateSkipsExistingVerdicts(t *testing.T) {
	server, store := sharedVerdictTestServer(t)
	var calls atomic.Int64
	upstream, lenient, strict := remoteVerdictMTLSTestServer(t, &calls)
	anonymousTLS := lenient
	anonymousTLS.CertFile, anonymousTLS.KeyFile = "", ""
	anonymous := func() *Classifier {
		return remoteVerdictTestClassifier(t, store, upstream.URL, "guard-v1", func(deployment *config.ModelDeployment) {
			deployment.TLS = &anonymousTLS
		})
	}
	text := scanProse(4_000)
	baseline := evaluateTurn(anonymous(), text)
	require.Empty(t, baseline.SignalErrors)
	require.False(t, guarded(baseline))
	require.NotEmpty(t, server.Keys(), "server-only TLS must still populate the shared cache")
	before := calls.Load()
	reused := evaluateTurn(anonymous(), text)
	require.Empty(t, reused.SignalErrors)
	require.Equal(t, baseline.SignalValues, reused.SignalValues)
	require.Equal(t, before, calls.Load(), "server-only TLS must still reuse shared windows")

	authenticated := remoteVerdictTestClassifier(t, store, upstream.URL, "guard-v1", func(deployment *config.ModelDeployment) {
		deployment.TLS = &strict
	})
	commands := server.CommandCount()
	before = calls.Load()
	got := evaluateTurn(authenticated, text)
	require.Empty(t, got.SignalErrors)
	require.True(t, guarded(got), "a client certificate must not inherit an anonymous verdict")
	require.Greater(t, calls.Load(), before)
	require.Empty(t, authenticated.windowVerdicts.entries)
	require.Equal(t, commands, server.CommandCount(), "client-authenticated deployments must skip Redis reads, writes and renewals")
}

// The endpoint verifies both client certificates and selects the strict policy
// by certificate identity. It also accepts anonymous TLS to seed old verdicts.
func remoteVerdictMTLSTestServer(t *testing.T, calls *atomic.Int64) (*httptest.Server, config.DeploymentTLS, config.DeploymentTLS) {
	t.Helper()
	lenient := connectortest.NewPrivateCA(t, "guard.example")
	strict := connectortest.NewPrivateCA(t, "guard.example")
	clients := lenient.ClientPool.Clone()
	chain, err := os.ReadFile(strict.ChainFile)
	require.NoError(t, err)
	require.True(t, clients.AppendCertsFromPEM(chain))
	strictCertificate, err := tls.LoadX509KeyPair(strict.ClientCert, strict.ClientKey)
	require.NoError(t, err)
	server := httptest.NewUnstartedServer(remoteVerdictTestHandler(calls, func(r *http.Request) bool {
		return len(r.TLS.PeerCertificates) > 0 && bytes.Equal(r.TLS.PeerCertificates[0].Raw, strictCertificate.Certificate[0])
	}))
	server.TLS = &tls.Config{
		MinVersion:   tls.VersionTLS12,
		Certificates: []tls.Certificate{lenient.Server},
		ClientAuth:   tls.VerifyClientCertIfGiven,
		ClientCAs:    clients,
	}
	server.StartTLS()
	t.Cleanup(server.Close)
	connection := config.DeploymentTLS{
		CAFile: lenient.ChainFile, ServerName: lenient.ServerName,
		CertFile: lenient.ClientCert, KeyFile: lenient.ClientKey,
	}
	strictConnection := connection
	strictConnection.CertFile, strictConnection.KeyFile = strict.ClientCert, strict.ClientKey
	return server, connection, strictConnection
}
