package config

import "testing"

func TestServedModelPrefersTheArtifact(t *testing.T) {
	for _, tc := range []struct {
		deployment ModelDeployment
		want       string
	}{
		{ModelDeployment{Provider: ModelRuntimeProvider, Artifact: "vllm-sr/Vela-1.0-Encoder-307M-Domain"}, "vllm-sr/Vela-1.0-Encoder-307M-Domain"},
		{ModelDeployment{Provider: ModelRuntimeProvider, Artifact: "/models/domain", Endpoint: "http://runtime:8100"}, "/models/domain"},
		{ModelDeployment{Provider: ModelRuntimeProvider, Endpoint: "http://runtime:8100"}, "domain-a"},
		{ModelDeployment{Provider: ModelRuntimeProvider, Endpoint: "http://runtime:8100", ServedName: "vela-domain"}, "vela-domain"},
		{ModelDeployment{Provider: "http", Endpoint: "http://classifier:9000"}, ""},
	} {
		if got := tc.deployment.ServedModel("domain-a"); got != tc.want {
			t.Errorf("%+v: ServedModel = %q, want %q", tc.deployment, got, tc.want)
		}
	}
}
