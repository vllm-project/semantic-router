package v1alpha1

import "testing"

func TestValidateAutoscaling(t *testing.T) {
	tests := []struct {
		name    string
		sr      *SemanticRouter
		wantErr bool
	}{
		{
			name: "valid autoscaling",
			sr: &SemanticRouter{
				Spec: SemanticRouterSpec{
					Autoscaling: AutoscalingSpec{
						MinReplicas:                    int32Ptr(2),
						MaxReplicas:                    int32Ptr(10),
						TargetCPUUtilizationPercentage: int32Ptr(80),
					},
				},
			},
			wantErr: false,
		},
		{
			name: "minReplicas greater than maxReplicas",
			sr: &SemanticRouter{
				Spec: SemanticRouterSpec{
					Autoscaling: AutoscalingSpec{
						MinReplicas:                    int32Ptr(10),
						MaxReplicas:                    int32Ptr(2),
						TargetCPUUtilizationPercentage: int32Ptr(80),
					},
				},
			},
			wantErr: true,
		},
		{
			name: "no metrics specified",
			sr: &SemanticRouter{
				Spec: SemanticRouterSpec{
					Autoscaling: AutoscalingSpec{
						MinReplicas: int32Ptr(2),
						MaxReplicas: int32Ptr(10),
					},
				},
			},
			wantErr: true,
		},
		{
			name: "CPU metric only",
			sr: &SemanticRouter{
				Spec: SemanticRouterSpec{
					Autoscaling: AutoscalingSpec{
						TargetCPUUtilizationPercentage: int32Ptr(80),
					},
				},
			},
			wantErr: false,
		},
		{
			name: "Memory metric only",
			sr: &SemanticRouter{
				Spec: SemanticRouterSpec{
					Autoscaling: AutoscalingSpec{
						TargetMemoryUtilizationPercentage: int32Ptr(80),
					},
				},
			},
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.sr.validateAutoscaling()
			if (err != nil) != tt.wantErr {
				t.Errorf("validateAutoscaling() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}
