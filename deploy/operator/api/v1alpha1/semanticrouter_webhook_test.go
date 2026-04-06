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
	"context"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestValidateCreate(t *testing.T) {
	pathType := ingressPathTypePrefix()

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
					Replicas: int32Ptr(2),
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
						Enabled:                        boolPtr(true),
						MinReplicas:                    int32Ptr(2),
						MaxReplicas:                    int32Ptr(10),
						TargetCPUUtilizationPercentage: int32Ptr(80),
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
						Enabled:     boolPtr(true),
						MinReplicas: int32Ptr(10),
						MaxReplicas: int32Ptr(2),
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
						Enabled:          boolPtr(true),
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
						Enabled: boolPtr(true),
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
					Replicas: int32Ptr(2),
					Autoscaling: AutoscalingSpec{
						Enabled:                        boolPtr(true),
						MinReplicas:                    int32Ptr(2),
						MaxReplicas:                    int32Ptr(10),
						TargetCPUUtilizationPercentage: int32Ptr(80),
					},
					Persistence: PersistenceSpec{
						Enabled:          boolPtr(true),
						Size:             "10Gi",
						StorageClassName: "standard",
					},
					Ingress: IngressSpec{
						Enabled: boolPtr(true),
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
						Enabled:          boolPtr(true),
						TimeoutSeconds:   int32Ptr(5),
						PeriodSeconds:    int32Ptr(10),
						FailureThreshold: int32Ptr(3),
					},
				},
			},
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := tt.sr.ValidateCreate(context.Background(), tt.sr)
			if (err != nil) != tt.wantErr {
				t.Errorf("ValidateCreate() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestValidateUpdate(t *testing.T) {
	pathType := ingressPathTypePrefix()

	old := &SemanticRouter{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test",
			Namespace: "default",
		},
		Spec: SemanticRouterSpec{
			Replicas: int32Ptr(1),
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
					Replicas: int32Ptr(3),
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
						Enabled:     boolPtr(true),
						MinReplicas: int32Ptr(10),
						MaxReplicas: int32Ptr(2),
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
						Enabled: boolPtr(true),
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
			_, err := tt.sr.ValidateUpdate(context.Background(), old, tt.sr)
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

	_, err := sr.ValidateDelete(context.Background(), sr)
	if err != nil {
		t.Errorf("ValidateDelete() should always succeed, got error: %v", err)
	}
}
