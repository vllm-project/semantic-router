package v1alpha1

import "testing"

type ingressValidationCase struct {
	name    string
	sr      *SemanticRouter
	wantErr bool
}

func ingressValidationCases(pathType string) []ingressValidationCase {
	return []ingressValidationCase{
		{
			name: "valid ingress",
			sr: &SemanticRouter{
				Spec: SemanticRouterSpec{
					Ingress: IngressSpec{
						Hosts: []IngressHost{
							{
								Host:  "example.com",
								Paths: []IngressPath{{Path: "/", PathType: pathType}},
							},
						},
					},
				},
			},
			wantErr: false,
		},
		{
			name:    "no hosts",
			sr:      &SemanticRouter{Spec: SemanticRouterSpec{Ingress: IngressSpec{Hosts: []IngressHost{}}}},
			wantErr: true,
		},
		{
			name: "empty host",
			sr: &SemanticRouter{
				Spec: SemanticRouterSpec{
					Ingress: IngressSpec{
						Hosts: []IngressHost{{Host: "", Paths: []IngressPath{{Path: "/", PathType: pathType}}}},
					},
				},
			},
			wantErr: true,
		},
		{
			name: "no paths",
			sr: &SemanticRouter{
				Spec: SemanticRouterSpec{
					Ingress: IngressSpec{Hosts: []IngressHost{{Host: "example.com", Paths: []IngressPath{}}}},
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
							{Host: "example.com", Paths: []IngressPath{{Path: "/", PathType: pathType}}},
							{Host: "api.example.com", Paths: []IngressPath{{Path: "/api", PathType: pathType}}},
						},
					},
				},
			},
			wantErr: false,
		},
	}
}

func TestValidateIngress(t *testing.T) {
	pathType := ingressPathTypePrefix()

	for _, tt := range ingressValidationCases(pathType) {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.sr.validateIngress()
			if (err != nil) != tt.wantErr {
				t.Errorf("validateIngress() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}
