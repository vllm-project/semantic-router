package controllers

import (
	"context"
	"testing"

	"gopkg.in/yaml.v3"

	vllmv1alpha1 "github.com/vllm-project/semantic-router/operator/api/v1alpha1"
	routerconfig "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// A gateway that sends request bodies in STREAMED or FullDuplexStreamed mode
// needs the Router's streamed body handler; the CR must be able to enable it.
func TestOperatorStreamedBodyReachesParsedRouterConfig(t *testing.T) {
	cases := []struct {
		name        string
		spec        *vllmv1alpha1.StreamedBodyConfig
		wantEnabled bool
		wantBytes   int64
		wantTimeout int
	}{
		{name: "omitted keeps buffered handling"},
		{
			name:        "enabled with limits",
			spec:        &vllmv1alpha1.StreamedBodyConfig{Enabled: true, MaxBytes: 10485760, TimeoutSec: 30},
			wantEnabled: true, wantBytes: 10485760, wantTimeout: 30,
		},
		{
			name:        "enabled without limits",
			spec:        &vllmv1alpha1.StreamedBodyConfig{Enabled: true},
			wantEnabled: true,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			r := &SemanticRouterReconciler{}
			sr := &vllmv1alpha1.SemanticRouter{Spec: vllmv1alpha1.SemanticRouterSpec{
				Config: vllmv1alpha1.ConfigSpec{StreamedBody: tc.spec},
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
			if cfg.StreamedBodyMode != tc.wantEnabled ||
				cfg.MaxStreamedBodyBytes != tc.wantBytes ||
				cfg.StreamedBodyTimeoutSec != tc.wantTimeout {
				t.Fatalf("streamed body = {enabled:%v max_bytes:%d timeout_sec:%d}, want {%v %d %d}",
					cfg.StreamedBodyMode, cfg.MaxStreamedBodyBytes, cfg.StreamedBodyTimeoutSec,
					tc.wantEnabled, tc.wantBytes, tc.wantTimeout)
			}
		})
	}
}
