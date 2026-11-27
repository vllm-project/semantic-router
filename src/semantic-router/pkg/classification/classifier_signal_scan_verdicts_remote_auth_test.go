package classification

import (
	"net/http"
	"os"
	"path/filepath"
	"sync/atomic"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// Static credentials can select a guard policy independently of its endpoint
// and model. The same credential source can also rotate between generations.
func TestSharedGuardWindowsStaticAuthChange(t *testing.T) {
	for _, source := range []string{"environment", "file"} {
		t.Run(source, func(t *testing.T) {
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
					auth := &config.DeploymentAuth{}
					setKey := func(key string) {
						t.Setenv("GUARD_VERDICT_TEST_API_KEY", key)
					}
					if source == "file" {
						auth.APIKeyFile = filepath.Join(t.TempDir(), "guard-key")
						setKey = func(key string) {
							require.NoError(t, os.WriteFile(auth.APIKeyFile, []byte(key), 0o600))
						}
					} else {
						auth.APIKeyEnv = "GUARD_VERDICT_TEST_API_KEY"
					}
					var calls atomic.Int64
					upstream := remoteVerdictTestServer(t, &calls, func(r *http.Request) bool {
						return r.Header.Get("Authorization") == "Bearer strict-test-key"
					})
					text := scanProse(4_000)
					var classifiers []*Classifier
					for _, key := range []string{"lenient-test-key", "strict-test-key", "lenient-test-key"} {
						setKey(key)
						classifier := remoteVerdictTestClassifier(t, store, upstream.URL, "guard-v1", func(deployment *config.ModelDeployment) {
							deployment.Auth = auth
						})
						classifiers = append(classifiers, classifier)
						before := calls.Load()
						got := evaluateTurn(classifier, text)
						require.Empty(t, got.SignalErrors)
						require.Equal(t, key == "strict-test-key", guarded(got), "a changed credential must not inherit another policy's verdict")
						require.Greater(t, calls.Load(), before, "each generation must ask the guard with its own credential")
					}
					for _, classifier := range classifiers {
						require.Empty(t, classifier.windowVerdicts.entries, "authenticated deployments must not retain window verdicts")
					}
					require.Empty(t, server.Keys(), "authenticated deployments must not publish window verdicts to Redis")
				})
			}
		})
	}
}

func TestSharedGuardWindowsStaticAuthSkipsExistingVerdicts(t *testing.T) {
	server, store := sharedVerdictTestServer(t)
	var calls atomic.Int64
	upstream := remoteVerdictTestServer(t, &calls, func(r *http.Request) bool {
		return r.Header.Get("X-Policy-Key") == "strict-test-key"
	})
	text := scanProse(4_000)
	anonymous := remoteVerdictTestClassifier(t, store, upstream.URL, "guard-v1")
	baseline := evaluateTurn(anonymous, text)
	require.Empty(t, baseline.SignalErrors)
	require.False(t, guarded(baseline))
	keys := server.Keys()
	require.NotEmpty(t, keys, "the anonymous deployment must populate the shared cache")

	t.Setenv("GUARD_VERDICT_TEST_API_KEY", "strict-test-key")
	authenticated := remoteVerdictTestClassifier(t, store, upstream.URL, "guard-v1", func(deployment *config.ModelDeployment) {
		prefix := ""
		deployment.Auth = &config.DeploymentAuth{
			APIKeyEnv: "GUARD_VERDICT_TEST_API_KEY", Header: "X-Policy-Key", Prefix: &prefix,
		}
	})
	before := calls.Load()
	got := evaluateTurn(authenticated, text)
	require.Empty(t, got.SignalErrors)
	require.True(t, guarded(got), "an authenticated deployment must not reuse an anonymous verdict")
	require.Greater(t, calls.Load(), before)
	require.Empty(t, authenticated.windowVerdicts.entries)
	require.ElementsMatch(t, keys, server.Keys(), "the authenticated deployment must not publish window verdicts")
}
