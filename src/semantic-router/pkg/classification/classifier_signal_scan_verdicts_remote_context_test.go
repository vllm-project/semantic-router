package classification

import (
	"context"
	"net/http"
	"sync/atomic"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/principal"
)

// The same model can apply different policies to identical input depending on
// the caller. Both a warm classifier and another replica must ask it again.
func TestGuardWindowsRespectRemoteCallerContext(t *testing.T) {
	for _, test := range []struct {
		name      string
		header    string
		configure func(*config.ModelDeployment)
	}{
		{
			name: "delegated", header: "X-Guard-User",
			configure: func(deployment *config.ModelDeployment) {
				prefix := ""
				deployment.Auth = &config.DeploymentAuth{
					Credential: config.CredentialDelegated, Header: "X-Guard-User", Prefix: &prefix,
				}
			},
		},
		{
			name: "forwarded_headers", header: "X-User-Id",
			configure: func(deployment *config.ModelDeployment) {
				deployment.ForwardRequestHeaders = []string{"x-user-id"}
			},
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			for _, backend := range []string{"local", "redis"} {
				t.Run(backend, func(t *testing.T) {
					server, store := sharedVerdictTestServer(t)
					if backend == "local" {
						store = nil
					}
					var calls atomic.Int64
					upstream := remoteVerdictTestServer(t, &calls, func(r *http.Request) bool {
						return r.Header.Get(test.header) == "bob"
					})
					first := remoteVerdictTestClassifier(t, store, upstream.URL, "guard-v1", test.configure)
					second := remoteVerdictTestClassifier(t, store, upstream.URL, "guard-v1", test.configure)
					text := scanProse(4_000)
					for _, request := range []struct {
						classifier *Classifier
						caller     string
					}{
						{first, "alice"},
						{second, "bob"},
						{first, "bob"},
						{second, "alice"},
						{first, "alice"},
					} {
						before := calls.Load()
						ctx := principal.ForRequestWithHeaders(context.Background(), request.caller,
							map[string]string{"x-user-id": request.caller})
						got := request.classifier.evaluateAllSignalsWithContext(SignalEvaluationInput{
							Text: text, CurrentUserText: text, RequestFacts: RequestFacts{Context: ctx},
						}, request.classifier.Config.Decisions, true)
						require.Empty(t, got.SignalErrors)
						require.Equal(t, request.caller == "bob", guarded(got), "verdict must use this caller's policy")
						require.Greater(t, calls.Load(), before, "every request must reach the remote guard")
					}
					require.Empty(t, first.windowVerdicts.entries, "caller-dependent verdicts must not be retained locally")
					require.Empty(t, second.windowVerdicts.entries)
					require.Empty(t, server.Keys(), "caller-dependent verdicts must not be published to Redis")
				})
			}
		})
	}
}
