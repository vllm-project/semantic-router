package v1alpha1

import "testing"

func TestValidateProbes(t *testing.T) {
	tests := []struct {
		name    string
		sr      *SemanticRouter
		wantErr bool
	}{
		{
			name: "valid probe",
			sr: &SemanticRouter{
				Spec: SemanticRouterSpec{
					StartupProbe: &ProbeSpec{
						Enabled:          boolPtr(true),
						TimeoutSeconds:   int32Ptr(5),
						PeriodSeconds:    int32Ptr(10),
						FailureThreshold: int32Ptr(3),
					},
				},
			},
			wantErr: false,
		},
		{
			name: "invalid timeout",
			sr: &SemanticRouter{
				Spec: SemanticRouterSpec{
					StartupProbe: &ProbeSpec{
						Enabled:        boolPtr(true),
						TimeoutSeconds: int32Ptr(0),
					},
				},
			},
			wantErr: true,
		},
		{
			name: "invalid period",
			sr: &SemanticRouter{
				Spec: SemanticRouterSpec{
					LivenessProbe: &ProbeSpec{
						Enabled:       boolPtr(true),
						PeriodSeconds: int32Ptr(-1),
					},
				},
			},
			wantErr: true,
		},
		{
			name: "invalid failure threshold",
			sr: &SemanticRouter{
				Spec: SemanticRouterSpec{
					ReadinessProbe: &ProbeSpec{
						Enabled:          boolPtr(true),
						FailureThreshold: int32Ptr(0),
					},
				},
			},
			wantErr: true,
		},
		{
			name: "probe disabled",
			sr: &SemanticRouter{
				Spec: SemanticRouterSpec{
					StartupProbe: &ProbeSpec{
						Enabled:        boolPtr(false),
						TimeoutSeconds: int32Ptr(-5),
					},
				},
			},
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.sr.validateProbes()
			if (err != nil) != tt.wantErr {
				t.Errorf("validateProbes() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}
