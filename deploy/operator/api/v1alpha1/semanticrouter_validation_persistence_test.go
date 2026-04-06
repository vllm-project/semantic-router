package v1alpha1

import "testing"

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
						Enabled:          boolPtr(true),
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
						Enabled:          boolPtr(true),
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
						Enabled:       boolPtr(true),
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
						Enabled:          boolPtr(false),
						ExistingClaim:    "my-claim",
						StorageClassName: "standard",
					},
				},
			},
			wantErr: false,
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
