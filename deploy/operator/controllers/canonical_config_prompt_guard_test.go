package controllers

import (
	"context"
	"testing"

	vllmv1alpha1 "github.com/vllm-project/semantic-router/operator/api/v1alpha1"
)

func TestBuildCanonicalConfigPromptGuardMappingPath(t *testing.T) {
	const defaultMappingPath = "models/mmbert32k-jailbreak-detector-merged/jailbreak_type_mapping.json"

	tests := []struct {
		name        string
		enabled     bool
		mappingPath string
		want        string
	}{
		{
			name:    "enabled guard inherits router default",
			enabled: true,
			want:    defaultMappingPath,
		},
		{
			name:        "enabled guard preserves explicit override",
			enabled:     true,
			mappingPath: "/config/custom-jailbreak-mapping.json",
			want:        "/config/custom-jailbreak-mapping.json",
		},
		{
			name: "disabled guard stays disabled without a mapping",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			r := &SemanticRouterReconciler{}
			sr := &vllmv1alpha1.SemanticRouter{
				Spec: vllmv1alpha1.SemanticRouterSpec{
					Config: vllmv1alpha1.ConfigSpec{
						PromptGuard: &vllmv1alpha1.PromptGuardConfig{
							Enabled:              tt.enabled,
							Threshold:            "0.7",
							JailbreakMappingPath: tt.mappingPath,
						},
					},
				},
			}

			canonical, err := r.buildCanonicalConfig(context.Background(), sr)
			if err != nil {
				t.Fatalf("buildCanonicalConfig failed: %v", err)
			}

			got := canonical.Global.ModelCatalog.Modules.PromptGuard.JailbreakMappingPath
			if got != tt.want {
				t.Fatalf("jailbreak_mapping_path = %q, want %q", got, tt.want)
			}
		})
	}
}
