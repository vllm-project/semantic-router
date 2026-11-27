package classification

import (
	"net/http"
	"sync/atomic"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// A static header can select the guard policy even when its endpoint and model
// stay the same. Independent classifier generations must not share that policy's
// answers after the header changes.
func TestSharedGuardWindowsStaticHeaderChange(t *testing.T) {
	for _, change := range []string{"value", "added", "removed"} {
		t.Run(change, func(t *testing.T) {
			for _, shared := range []bool{false, true} {
				name := "local"
				if shared {
					name = "shared"
				}
				t.Run(name, func(t *testing.T) {
					_, store := sharedVerdictTestServer(t)
					if !shared {
						store = nil
					}
					var calls atomic.Int64
					upstream := remoteVerdictTestServer(t, &calls, func(r *http.Request) bool {
						policy := r.Header.Get("X-Policy")
						return policy == "strict" || (change == "removed" && policy == "")
					})
					headers := [][2]string{{"X-Tenant", "tenant-a"}, {"X-Policy", "lenient"}}
					if change == "added" {
						headers = headers[:1]
					}
					classifier := func(fields [][2]string) *Classifier {
						return remoteVerdictTestClassifier(t, store, upstream.URL, "guard-v1", func(deployment *config.ModelDeployment) {
							deployment.Headers = make(map[string]string, len(fields))
							for _, field := range fields {
								deployment.Headers[field[0]] = field[1]
							}
						})
					}
					text := scanProse(4_000)
					baseline := evaluateTurn(classifier(headers), text)
					require.Empty(t, baseline.SignalErrors)
					require.False(t, guarded(baseline))
					require.Positive(t, calls.Load())

					// Reconstruct the same headers in reverse insertion order in a
					// new replica: a map's iteration order must not affect reuse.
					reversed := make([][2]string, len(headers))
					for i, field := range headers {
						reversed[len(headers)-1-i] = field
					}
					before := calls.Load()
					got := evaluateTurn(classifier(reversed), text)
					require.Empty(t, got.SignalErrors)
					require.Equal(t, baseline.SignalValues, got.SignalValues)
					require.Equal(t, baseline.ScanCoverage, got.ScanCoverage)
					if shared {
						require.Equal(t, before, calls.Load(), "unchanged static headers must allow shared reuse")
					} else {
						require.Greater(t, calls.Load(), before)
					}

					headers = [][2]string{{"X-Tenant", "tenant-a"}, {"X-Policy", "strict"}}
					if change == "removed" {
						headers = headers[:1]
					}
					before = calls.Load()
					got = evaluateTurn(classifier(headers), text)
					require.Empty(t, got.SignalErrors)
					require.True(t, guarded(got), "a changed static header must not inherit the earlier benign verdict")
					require.Greater(t, calls.Load(), before, "the guard must evaluate the new header policy")
				})
			}
		})
	}
}
