/*
Copyright 2026 vLLM Semantic Router Contributors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package v1alpha1

import (
	"testing"

	networkingv1 "k8s.io/api/networking/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

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
						MinReplicas:                    func() *int32 { i := int32(2); return &i }(),
						MaxReplicas:                    func() *int32 { i := int32(10); return &i }(),
						TargetCPUUtilizationPercentage: func() *int32 { i := int32(80); return &i }(),
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
						MinReplicas:                    func() *int32 { i := int32(10); return &i }(),
						MaxReplicas:                    func() *int32 { i := int32(2); return &i }(),
						TargetCPUUtilizationPercentage: func() *int32 { i := int32(80); return &i }(),
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
						MinReplicas: func() *int32 { i := int32(2); return &i }(),
						MaxReplicas: func() *int32 { i := int32(10); return &i }(),
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
						TargetCPUUtilizationPercentage: func() *int32 { i := int32(80); return &i }(),
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
						TargetMemoryUtilizationPercentage: func() *int32 { i := int32(80); return &i }(),
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

func TestValidatePersistence(t *testing.T) {
	tests := []struct {
		name    string
		sr      *SemanticRouter
		wantErr bool
	}{
		{
			name: "valid persistence",
			sr: &SemanticRouter{
				Spec: SemanticRouterSpec{
					Persistence: PersistenceSpec{
						Enabled:          func() *bool { b := true; return &b }(),
						Size:             "10Gi",
						StorageClassName: "standard",
					},
				},
			},
			wantErr: false,
		},
		{
			name: "both existingClaim and storageClassName",
			sr: &SemanticRouter{
				Spec: SemanticRouterSpec{
					Persistence: PersistenceSpec{
						Enabled:          func() *bool { b := true; return &b }(),
						ExistingClaim:    "my-claim",
						StorageClassName: "standard",
					},
				},
			},
			wantErr: true,
		},
		{
			name: "existingClaim only",
			sr: &SemanticRouter{
				Spec: SemanticRouterSpec{
					Persistence: PersistenceSpec{
						Enabled:       func() *bool { b := true; return &b }(),
						ExistingClaim: "my-claim",
					},
				},
			},
			wantErr: false,
		},
		{
			name: "persistence disabled",
			sr: &SemanticRouter{
				Spec: SemanticRouterSpec{
					Persistence: PersistenceSpec{
						Enabled:          func() *bool { b := false; return &b }(),
						ExistingClaim:    "my-claim",
						StorageClassName: "standard",
					},
				},
			},
			wantErr: false, // Should not validate when disabled
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.sr.validatePersistence()
			if (err != nil) != tt.wantErr {
				t.Errorf("validatePersistence() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

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
						Enabled:          func() *bool { b := true; return &b }(),
						TimeoutSeconds:   func() *int32 { i := int32(5); return &i }(),
						PeriodSeconds:    func() *int32 { i := int32(10); return &i }(),
						FailureThreshold: func() *int32 { i := int32(3); return &i }(),
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
						Enabled:        func() *bool { b := true; return &b }(),
						TimeoutSeconds: func() *int32 { i := int32(0); return &i }(),
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
						Enabled:       func() *bool { b := true; return &b }(),
						PeriodSeconds: func() *int32 { i := int32(-1); return &i }(),
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
						Enabled:          func() *bool { b := true; return &b }(),
						FailureThreshold: func() *int32 { i := int32(0); return &i }(),
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
						Enabled:        func() *bool { b := false; return &b }(),
						TimeoutSeconds: func() *int32 { i := int32(-5); return &i }(), // Invalid but disabled
					},
				},
			},
			wantErr: false, // Should not validate when disabled
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

func TestValidateIngress(t *testing.T) {
	pathType := string(networkingv1.PathTypePrefix)

	tests := []struct {
		name    string
		sr      *SemanticRouter
		wantErr bool
	}{
		{
			name: "valid ingress",
			sr: &SemanticRouter{
				Spec: SemanticRouterSpec{
					Ingress: IngressSpec{
						Hosts: []IngressHost{
							{
								Host: "example.com",
								Paths: []IngressPath{
									{
										Path:     "/",
										PathType: pathType,
									},
								},
							},
						},
					},
				},
			},
			wantErr: false,
		},
		{
			name: "no hosts",
			sr: &SemanticRouter{
				Spec: SemanticRouterSpec{
					Ingress: IngressSpec{
						Hosts: []IngressHost{},
					},
				},
			},
			wantErr: true,
		},
		{
			name: "empty host",
			sr: &SemanticRouter{
				Spec: SemanticRouterSpec{
					Ingress: IngressSpec{
						Hosts: []IngressHost{
							{
								Host: "",
								Paths: []IngressPath{
									{
										Path:     "/",
										PathType: pathType,
									},
								},
							},
						},
					},
				},
			},
			wantErr: true,
		},
		{
			name: "no paths",
			sr: &SemanticRouter{
				Spec: SemanticRouterSpec{
					Ingress: IngressSpec{
						Hosts: []IngressHost{
							{
								Host:  "example.com",
								Paths: []IngressPath{},
							},
						},
					},
				},
			},
			wantErr: true,
		},
		{
			name: "multiple hosts",
			sr: &SemanticRouter{
				Spec: SemanticRouterSpec{
					Ingress: IngressSpec{
						Hosts: []IngressHost{
							{
								Host: "example.com",
								Paths: []IngressPath{
									{Path: "/", PathType: pathType},
								},
							},
							{
								Host: "api.example.com",
								Paths: []IngressPath{
									{Path: "/api", PathType: pathType},
								},
							},
						},
					},
				},
			},
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.sr.validateIngress()
			if (err != nil) != tt.wantErr {
				t.Errorf("validateIngress() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestValidateCreate(t *testing.T) {
	pathType := string(networkingv1.PathTypePrefix)

	tests := []struct {
		name    string
		sr      *SemanticRouter
		wantErr bool
	}{
		{
			name: "valid semantic router",
			sr: &SemanticRouter{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test",
					Namespace: "default",
				},
				Spec: SemanticRouterSpec{
					Replicas: func() *int32 { i := int32(2); return &i }(),
				},
			},
			wantErr: false,
		},
		{
			name: "valid with autoscaling",
			sr: &SemanticRouter{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test",
					Namespace: "default",
				},
				Spec: SemanticRouterSpec{
					Autoscaling: AutoscalingSpec{
						Enabled:                        func() *bool { b := true; return &b }(),
						MinReplicas:                    func() *int32 { i := int32(2); return &i }(),
						MaxReplicas:                    func() *int32 { i := int32(10); return &i }(),
						TargetCPUUtilizationPercentage: func() *int32 { i := int32(80); return &i }(),
					},
				},
			},
			wantErr: false,
		},
		{
			name: "invalid autoscaling",
			sr: &SemanticRouter{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test",
					Namespace: "default",
				},
				Spec: SemanticRouterSpec{
					Autoscaling: AutoscalingSpec{
						Enabled:     func() *bool { b := true; return &b }(),
						MinReplicas: func() *int32 { i := int32(10); return &i }(),
						MaxReplicas: func() *int32 { i := int32(2); return &i }(),
					},
				},
			},
			wantErr: true,
		},
		{
			name: "invalid persistence",
			sr: &SemanticRouter{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test",
					Namespace: "default",
				},
				Spec: SemanticRouterSpec{
					Persistence: PersistenceSpec{
						Enabled:          func() *bool { b := true; return &b }(),
						ExistingClaim:    "claim",
						StorageClassName: "standard",
					},
				},
			},
			wantErr: true,
		},
		{
			name: "invalid ingress",
			sr: &SemanticRouter{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test",
					Namespace: "default",
				},
				Spec: SemanticRouterSpec{
					Ingress: IngressSpec{
						Enabled: func() *bool { b := true; return &b }(),
						Hosts:   []IngressHost{},
					},
				},
			},
			wantErr: true,
		},
		{
			name: "valid complete configuration",
			sr: &SemanticRouter{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test",
					Namespace: "default",
				},
				Spec: SemanticRouterSpec{
					Replicas: func() *int32 { i := int32(2); return &i }(),
					Autoscaling: AutoscalingSpec{
						Enabled:                        func() *bool { b := true; return &b }(),
						MinReplicas:                    func() *int32 { i := int32(2); return &i }(),
						MaxReplicas:                    func() *int32 { i := int32(10); return &i }(),
						TargetCPUUtilizationPercentage: func() *int32 { i := int32(80); return &i }(),
					},
					Persistence: PersistenceSpec{
						Enabled:          func() *bool { b := true; return &b }(),
						Size:             "10Gi",
						StorageClassName: "standard",
					},
					Ingress: IngressSpec{
						Enabled: func() *bool { b := true; return &b }(),
						Hosts: []IngressHost{
							{
								Host: "example.com",
								Paths: []IngressPath{
									{Path: "/", PathType: pathType},
								},
							},
						},
					},
					StartupProbe: &ProbeSpec{
						Enabled:          func() *bool { b := true; return &b }(),
						TimeoutSeconds:   func() *int32 { i := int32(5); return &i }(),
						PeriodSeconds:    func() *int32 { i := int32(10); return &i }(),
						FailureThreshold: func() *int32 { i := int32(3); return &i }(),
					},
				},
			},
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := tt.sr.ValidateCreate()
			if (err != nil) != tt.wantErr {
				t.Errorf("ValidateCreate() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestValidateUpdate(t *testing.T) {
	pathType := string(networkingv1.PathTypePrefix)

	old := &SemanticRouter{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test",
			Namespace: "default",
		},
		Spec: SemanticRouterSpec{
			Replicas: func() *int32 { i := int32(1); return &i }(),
		},
	}

	tests := []struct {
		name    string
		sr      *SemanticRouter
		wantErr bool
	}{
		{
			name: "valid update",
			sr: &SemanticRouter{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test",
					Namespace: "default",
				},
				Spec: SemanticRouterSpec{
					Replicas: func() *int32 { i := int32(3); return &i }(),
				},
			},
			wantErr: false,
		},
		{
			name: "invalid autoscaling update",
			sr: &SemanticRouter{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test",
					Namespace: "default",
				},
				Spec: SemanticRouterSpec{
					Autoscaling: AutoscalingSpec{
						Enabled:     func() *bool { b := true; return &b }(),
						MinReplicas: func() *int32 { i := int32(10); return &i }(),
						MaxReplicas: func() *int32 { i := int32(2); return &i }(),
					},
				},
			},
			wantErr: true,
		},
		{
			name: "enable ingress",
			sr: &SemanticRouter{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test",
					Namespace: "default",
				},
				Spec: SemanticRouterSpec{
					Ingress: IngressSpec{
						Enabled: func() *bool { b := true; return &b }(),
						Hosts: []IngressHost{
							{
								Host: "example.com",
								Paths: []IngressPath{
									{Path: "/", PathType: pathType},
								},
							},
						},
					},
				},
			},
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := tt.sr.ValidateUpdate(old)
			if (err != nil) != tt.wantErr {
				t.Errorf("ValidateUpdate() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestValidateDelete(t *testing.T) {
	sr := &SemanticRouter{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test",
			Namespace: "default",
		},
		Spec: SemanticRouterSpec{},
	}

	_, err := sr.ValidateDelete()
	if err != nil {
		t.Errorf("ValidateDelete() should always succeed, got error: %v", err)
	}
}
