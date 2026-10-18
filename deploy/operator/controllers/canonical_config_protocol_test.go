package controllers

import (
	"context"
	"testing"

	"gopkg.in/yaml.v3"

	vllmv1alpha1 "github.com/vllm-project/semantic-router/operator/api/v1alpha1"
	routerconfig "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// Gateways in front of models that accept richer OpenAI Chat fields need the
// CR to opt in to preserving them; the CR must be able to enable it.
func TestOperatorProtocolReachesParsedRouterConfig(t *testing.T) {
	cases := []struct {
		name string
		spec *vllmv1alpha1.ProtocolConfig
		want bool
	}{
		{name: "omitted keeps the strict contract"},
		{
			name: "preserve enabled",
			spec: &vllmv1alpha1.ProtocolConfig{PreserveUnknownChatFields: true},
			want: true,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			r := &SemanticRouterReconciler{}
			sr := &vllmv1alpha1.SemanticRouter{Spec: vllmv1alpha1.SemanticRouterSpec{
				Config: vllmv1alpha1.ConfigSpec{Protocol: tc.spec},
			}}
			canonical, err := r.buildCanonicalConfig(context.Background(), sr)
			if err != nil {
				t.Fatalf("buildCanonicalConfig failed: %v", err)
			}
			data, err := yaml.Marshal(canonical)
			if err != nil {
				t.Fatalf("marshal generated config: %v", err)
			}
			cfg, err := routerconfig.ParseYAMLBytes(data)
			if err != nil {
				t.Fatalf("router rejects the operator-generated config: %v\n%s", err, data)
			}
			if cfg.PreserveUnknownChatFields != tc.want {
				t.Fatalf("preserve_unknown_chat_fields = %v, want %v", cfg.PreserveUnknownChatFields, tc.want)
			}
		})
	}
}
