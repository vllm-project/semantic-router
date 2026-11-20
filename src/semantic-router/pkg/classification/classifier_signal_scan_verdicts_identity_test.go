package classification

import (
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestGuardWindowCredentialsDisableEveryProvider(t *testing.T) {
	for _, provider := range []string{config.SystemOneProvider, config.ModelRuntimeProvider} {
		for name, configure := range map[string]func(*config.ModelDeployment){
			"static": func(d *config.ModelDeployment) { d.Auth = &config.DeploymentAuth{APIKeyEnv: "GUARD_KEY"} },
			"delegated": func(d *config.ModelDeployment) {
				d.Auth = &config.DeploymentAuth{Credential: config.CredentialDelegated}
			},
			"certificate": func(d *config.ModelDeployment) {
				d.TLS = &config.DeploymentTLS{CertFile: "/client.crt", KeyFile: "/client.key"}
			},
			"key_only":  func(d *config.ModelDeployment) { d.TLS = &config.DeploymentTLS{KeyFile: "/client.key"} },
			"forwarded": func(d *config.ModelDeployment) { d.ForwardRequestHeaders = []string{"X-User"} },
		} {
			t.Run(provider+"/"+name, func(t *testing.T) {
				d := config.ModelDeployment{Provider: provider, Endpoint: "https://guard.example", Model: "guard-v1"}
				configure(&d)
				cfg := &config.RouterConfig{}
				cfg.ModelDeployments = map[string]config.ModelDeployment{"guard": d}
				c := &Classifier{Config: cfg}
				_, ok := c.windowModelIdentity("guard")
				require.False(t, ok, "caller-dependent deployments must not get window cache identities")
			})
		}
	}
}

func TestGuardWindowMixedReplicaIdentity(t *testing.T) {
	cfg := &config.RouterConfig{}
	d := config.ModelDeployment{Provider: config.ModelRuntimeProvider, Artifact: "example/guard", Replicas: []config.ModelReplica{
		{Device: "cpu"}, {Endpoint: "http://one.example"}, {Endpoint: "http://two.example", Model: "guard-v1"},
	}}
	cfg.ModelDeployments = map[string]config.ModelDeployment{"guard": d}
	c := &Classifier{Config: cfg}
	before, ok := c.windowModelIdentity("guard")
	require.True(t, ok)
	d.Replicas[0], d.Replicas[2] = d.Replicas[2], d.Replicas[0]
	cfg.ModelDeployments["guard"] = d
	reordered, ok := c.windowModelIdentity("guard")
	require.True(t, ok)
	require.Equal(t, before, reordered)
	d.Replicas[1].Endpoint = "http://changed.example"
	cfg.ModelDeployments["guard"] = d
	changed, ok := c.windowModelIdentity("guard")
	require.True(t, ok)
	require.NotEqual(t, before, changed, "managed workers must not hide attached identity changes in the same pool")
}
