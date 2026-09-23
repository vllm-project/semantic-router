package handlers

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"testing"

	"github.com/vllm-project/semantic-router/dashboard/backend/auth"
)

type authorizationTestTopology struct {
	*fakeRuntimeTopology
	beforeInventory func()
}

func (topology *authorizationTestTopology) Inventory(ctx context.Context) (runtimeTopologyInventory, error) {
	if topology.beforeInventory != nil {
		topology.beforeInventory()
	}
	return topology.fakeRuntimeTopology.Inventory(ctx)
}

func TestRecipeMutationRejectsRevocationDuringOperation(t *testing.T) {
	tests := []struct {
		name, action, stage string
		revokeSession       bool
	}{
		{"activate preparation permission", "activate", "preparation", false},
		{"activate preparation session", "activate", "preparation", true},
		{"activate verification permission", "activate", "verification", false},
		{"activate verification session", "activate", "verification", true},
		{"deactivate preparation permission", "deactivate", "preparation", false},
		{"deactivate preparation session", "deactivate", "preparation", true},
		{"deactivate verification permission", "deactivate", "verification", false},
		{"deactivate verification session", "deactivate", "verification", true},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			store, summary, configPath := importedActivationFixture(t, "accuracy")
			authStore, err := auth.NewStore(filepath.Join(t.TempDir(), "auth.db"))
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() {
				if closeErr := authStore.Close(); closeErr != nil {
					t.Error(closeErr)
				}
			})
			svc := auth.NewService(authStore, "recipe-authorization-test-secret", 1)
			if bootstrapErr := svc.EnsureBootstrapAdmin(context.Background(), "recipe@example.com", "test-password", "Recipe Admin"); bootstrapErr != nil {
				t.Fatal(bootstrapErr)
			}
			token, user, err := svc.Login(context.Background(), "recipe@example.com", "test-password")
			if err != nil {
				t.Fatal(err)
			}

			topology := &authorizationTestTopology{fakeRuntimeTopology: testTopologyForHotSwitch()}
			applyCalls := 0
			activator := NewRecipeActivator(RecipeActivatorOptions{
				Store: store, ConfigPath: configPath, ConfigDir: filepath.Dir(configPath),
				Topology: topology,
				RealizeConfig: func(raw []byte, _ string) ([]byte, error) {
					return append(append([]byte(nil), raw...), []byte("\n# realized recipe\n")...), nil
				},
				ApplyRuntime:  func(path, _ string) (string, error) { applyCalls++; return path, nil },
				VerifyRuntime: func(context.Context, string) error { return nil },
				VerifyEnvoy:   func(context.Context) error { return nil },
			})
			if test.action == "deactivate" {
				mustActivateRecipe(t, activator, summary)
			}
			applyCalls = 0
			original := mustReadFile(t, configPath)
			originalPointer, originalState, err := store.ActivationStatus()
			if err != nil {
				t.Fatal(err)
			}
			revoked := false
			revoke := func() {
				if revoked {
					return
				}
				revoked = true
				if test.revokeSession {
					if revokeErr := svc.RevokeToken(context.Background(), token); revokeErr != nil {
						t.Fatal(revokeErr)
					}
					return
				}
				if permissionErr := authStore.SetUserPermission(context.Background(), user.ID, auth.PermConfigDeploy, false); permissionErr != nil {
					t.Fatal(permissionErr)
				}
			}
			if test.stage == "preparation" {
				topology.beforeInventory = revoke
			} else {
				activator.verifyRuntime = func(context.Context, string) error { revoke(); return nil }
			}

			handler := NewRecipeHandler(nil, WithRecipePackages(store, activator, false))
			body := []byte(`{}`)
			if test.action == "activate" {
				body, err = json.Marshal(activationRequest(summary))
				if err != nil {
					t.Fatal(err)
				}
			}
			path := "/api/recipe/" + test.action
			routes := auth.NewPolicyMux()
			mutationHandler := handler.ActivatePackage
			if test.action == "deactivate" {
				mutationHandler = handler.DeactivatePackage
			}
			routes.HandleFunc(auth.ProtectedMutationRoute(path, auth.PermConfigDeploy, "recipe."+test.action,
				auth.SensitivitySecret, auth.ResourceOwnerConfig, 16<<20, http.MethodPost), mutationHandler)
			request := httptest.NewRequest(http.MethodPost, path, bytes.NewReader(body))
			request.Header.Set("Authorization", "Bearer "+token)
			response := httptest.NewRecorder()
			auth.AuthenticateRequest(svc, routes)(routes).ServeHTTP(response, request)
			wantStatus := http.StatusForbidden
			if test.revokeSession {
				wantStatus = http.StatusUnauthorized
			}
			if !revoked || response.Code != wantStatus {
				t.Fatalf("revoked=%v status=%d want=%d body=%s", revoked, response.Code, wantStatus, response.Body.String())
			}
			if !bytes.Equal(original, mustReadFile(t, configPath)) {
				t.Fatal("revoked mutation changed the runtime config")
			}
			pointer, state, err := store.ActivationStatus()
			if err != nil || state != originalState || pointer != originalPointer {
				t.Fatalf("activation status changed: pointer=%+v state=%s err=%v", pointer, state, err)
			}
			if test.stage == "preparation" && applyCalls != 0 {
				t.Fatalf("preparation rejection applied runtime %d times", applyCalls)
			}
			if test.stage == "verification" && applyCalls != 2 {
				t.Fatalf("verification rejection must apply then roll back, got %d calls", applyCalls)
			}
		})
	}
}
