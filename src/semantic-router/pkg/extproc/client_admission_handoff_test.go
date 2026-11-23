package extproc

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/clientauth"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/configsnapshot"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerruntime"
)

// Both the routing and engine-only publication paths must carry a live policy
// source. Pause in the actual swap callback, before publication, so this test
// cannot pass merely because the registry eventually catches up.
func TestClientAdmissionSharesPolicyDuringPublicationHandoff(t *testing.T) {
	t.Setenv("FAST_KEY", "fast-1")
	t.Setenv("FAST_KEY_NEXT", "fast-2")
	for _, withRouter := range []bool{true, false} {
		name := "router"
		if !withRouter {
			name = "engine"
		}
		t.Run(name, func(t *testing.T) {
			generation := func(keys []string) (*OpenAIRouter, *configsnapshot.Snapshot) {
				cfg, err := config.ParseYAMLBytes([]byte(clientAuthTestConfig))
				require.NoError(t, err)
				cfg.RouterOptions.RouterEnabled = &withRouter
				if len(keys) != 0 {
					cfg.Entrypoints[0].ClientAuth = &config.EntrypointClientAuth{Type: config.ClientAuthBearerAPIKey, APIKeyEnvs: keys}
				}
				snapshot, err := configsnapshot.NewManager(configsnapshot.Options{}).Install(context.Background(), configsnapshot.Update{Config: cfg})
				require.NoError(t, err)
				if !withRouter {
					return nil, snapshot
				}
				return &OpenAIRouter{Config: cfg}, snapshot
			}
			for _, tc := range []struct {
				name          string
				before, after []string
				want          clientauth.Decision
			}{
				{"protected", nil, []string{"FAST_KEY"}, clientauth.Superseded},
				{"rotated", []string{"FAST_KEY"}, []string{"FAST_KEY_NEXT"}, clientauth.Superseded},
				{"public", []string{"FAST_KEY"}, nil, clientauth.Superseded},
				{"unchanged", []string{"FAST_KEY"}, []string{"FAST_KEY"}, clientauth.Admitted},
				{"reordered", []string{"FAST_KEY", "FAST_KEY_NEXT"}, []string{"FAST_KEY_NEXT", "FAST_KEY"}, clientauth.Admitted},
			} {
				t.Run(tc.name, func(t *testing.T) {
					oldRouter, old := generation(tc.before)
					nextRouter, next := generation(tc.after)
					service := NewRouterServiceForSnapshot(oldRouter, old)
					t.Cleanup(func() { _ = service.Close() })
					registry := routerruntime.NewRegistry(nil)
					publishSnapshotState(old.Config(), old, oldRouter, registry, service.current.Load().acquire, service)
					lease, err := service.Pin()
					require.NoError(t, err)
					defer lease.Release()
					err = service.swapSnapshot(nextRouter, next, func(acquire AcquireFunc) {
						assert.Same(t, old, registry.ConfigSnapshot())
						assert.Same(t, next.ClientAuth(), lease.ActiveClientAuth())
						assert.Same(t, lease.ActiveClientAuth(), registry.ActiveClientAuth())
						if oldRouter != nil {
							_, active := oldRouter.clientPolicies()
							assert.Same(t, active, registry.ActiveClientAuth())
						}
						assert.Equal(t, tc.want, clientauth.Admit(old.ClientAuth(), registry.ActiveClientAuth(), config.ChatAPI, "company/fast", "Bearer fast-1"))
						publishSnapshotState(next.Config(), next, nextRouter, registry, acquire, service)
					})
					require.NoError(t, err)
					assert.Same(t, next, registry.ConfigSnapshot())
					assert.Same(t, next.ClientAuth(), registry.ActiveClientAuth())
				})
			}
		})
	}
}
