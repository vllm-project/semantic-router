package handlers

import (
	"testing"

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
			config := "version: v0.3\nglobal:\n  services:\n    management_api:\n" +
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
