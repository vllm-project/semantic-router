package handlers

import (
	"context"
	"net/http"
	"path/filepath"
	"testing"

	"github.com/vllm-project/semantic-router/dashboard/backend/auth"
	"github.com/vllm-project/semantic-router/dashboard/backend/recipe"
)

// A bearer Recipe may leave its token list to the Dashboard: activation binds
// the Dashboard's credential whether the list is missing, empty or names other
// callers, and keeps the callers it names.
func TestBearerBindingAddsTheDashboardCredential(t *testing.T) {
	for _, test := range []struct {
		name, tokens string
		want         int
	}{
		{name: "no_tokens", want: 1},
		{name: "empty_tokens", tokens: "        tokens: []\n", want: 1},
		{name: "other_caller", tokens: "        tokens:\n          - env: OPERATOR_TOKEN\n            role: admin\n", want: 2},
		{name: "other_role", tokens: "        tokens:\n          - env: " + recipe.ManagementCredentialEnv + "\n            role: viewer\n", want: 1},
	} {
		t.Run(test.name, func(t *testing.T) {
			config := "version: v0.5\nglobal:\n  services:\n    management_api:\n" +
				"      bind_address: 0.0.0.0\n      port: 8080\n      auth:\n        mode: bearer\n" + test.tokens
			bound, ok, err := bindActivationManagementCredential([]byte(config))
			if err != nil || !ok {
				t.Fatalf("bind = %v, %v", ok, err)
			}
			_, management, managed, err := activationManagementAPI(bound, true)
			if err != nil || !managed || len(management.Auth.Tokens) != test.want {
				t.Fatalf("managed = %v, %v; tokens = %+v, want %d\n%s", managed, err, management.Auth.Tokens, test.want, bound)
			}
		})
	}
}

// A Recipe whose config binds an environment reference, such as a provider key
// an auth.api_key_env names, makes the Router read a value the writer never
// sees and send it where the Recipe chose. A caller who may not read
// configuration secrets may not activate one; an admin may. Import gates on
// config.write, activation on config.deploy, so the Write role reaches both.
func TestActivatingAnEnvironmentBindingRecipeNeedsSecretsRead(t *testing.T) {
	store, summary, configPath := importedActivationFixture(t, "accuracy")
	activator := NewRecipeActivator(RecipeActivatorOptions{
		Store:         store,
		ConfigPath:    configPath,
		ConfigDir:     filepath.Dir(configPath),
		Topology:      testTopologyForHotSwitch(),
		RealizeConfig: func(raw []byte, _ string) ([]byte, error) { return raw, nil },
		ApplyRuntime:  func(path, _ string) (string, error) { return path, nil },
		VerifyRuntime: func(context.Context, string) error { return nil },
		VerifyEnvoy:   func(context.Context) error { return nil },
	})

	writer := auth.WithAuthContext(context.Background(), auth.AuthContext{Perms: map[string]bool{auth.PermConfigDeploy: true}})
	_, err := activator.Activate(writer, activationRequest(summary))
	packageErr, ok := recipe.AsPackageError(err)
	if !ok || packageErr.Status != http.StatusForbidden {
		t.Fatalf("a writer's activation: error = %#v, want 403", err)
	}
	if _, state, _ := store.ActivationStatus(); state != recipe.ActivationNone {
		t.Fatalf("a refused activation changed state to %q", state)
	}

	admin := auth.WithAuthContext(context.Background(), auth.AuthContext{Perms: map[string]bool{auth.PermConfigDeploy: true, auth.PermConfigSecretsRead: true}})
	result, err := activator.Activate(admin, activationRequest(summary))
	if err != nil || result.Status != "active" {
		t.Fatalf("an admin's activation: result = %#v, err = %v", result, err)
	}
}
