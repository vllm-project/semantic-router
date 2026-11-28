package recipe

import (
	"strings"
	"testing"
)

func TestNamedSecretBindingsRequireAuthorization(t *testing.T) {
	for _, field := range []string{"api_key_env", "key_env", "password_env", "hmac_key_env"} {
		t.Run(field, func(t *testing.T) {
			data := []byte("global:\n  stores:\n    guard_verdicts:\n      " + field + ": GUARD_SECRET\n")
			t.Setenv(recipeEnvironmentAllowlist, "")
			t.Setenv("GUARD_SECRET", "private-test-value")
			bound, err := HasEnvironmentBindings(data)
			if err != nil || !bound || len(detectEnvironmentBindingWarnings(data)) != 1 {
				t.Fatalf("binding not discovered: %v %v", bound, err)
			}
			if err := ValidateEnvironmentBindings(data); err == nil || strings.Contains(err.Error(), "private-test-value") {
				t.Fatalf("unauthorized binding: %v", err)
			}
			t.Setenv(recipeEnvironmentAllowlist, "GUARD_SECRET")
			if err := ValidateEnvironmentBindings(data); err != nil {
				t.Fatal(err)
			}
			for _, name := range []string{"PATH", "VLLM_SR_STACK_REDIS_PASSWORD", "VLLM_SR_STACK_POSTGRES_PASSWORD", ManagementCredentialEnv, "${GUARD_SECRET}", " GUARD_SECRET"} {
				if _, err := extractEnvironmentBindings([]byte(field + ": '" + name + "'\n")); err == nil {
					t.Fatalf("accepted reserved or invalid binding %s in %s", name, field)
				}
			}
		})
	}
}

func TestGuardBindingsDoNotRequireDashboardToHoldRouterSecrets(t *testing.T) {
	data := []byte("global:\n  stores:\n    guard_verdicts:\n      hmac_key_env: GUARD_SIGNING\n      redis:\n        password_env: GUARD_PASSWORD\n")
	t.Setenv("GUARD_SIGNING", "")
	t.Setenv("GUARD_PASSWORD", "")
	t.Setenv(recipeEnvironmentAllowlist, "GUARD_SIGNING")
	if err := ValidateEnvironmentBindings(data); err == nil {
		t.Fatal("Router-only password still requires authorization")
	}
	t.Setenv(recipeEnvironmentAllowlist, "GUARD_SIGNING,GUARD_PASSWORD")
	if err := ValidateEnvironmentBindings(data); err != nil {
		t.Fatal(err)
	}
	// The exception applies only to these named runtime credentials. Reusing
	// the name in interpolation does not bypass its availability check.
	data = append(data, []byte("value: ${GUARD_SIGNING}\n")...)
	if err := ValidateEnvironmentBindings(data); err == nil {
		t.Fatal("interpolation accepted an unavailable value")
	}
}

func TestNamedSecretBindingsCannotHideInYAMLAliases(t *testing.T) {
	t.Setenv(recipeEnvironmentAllowlist, "")
	for _, field := range []string{"api_key_env", "key_env", "password_env", "hmac_key_env"} {
		for _, data := range []string{
			"name: &secret GUARD_SECRET\nstore: {" + field + ": *secret}\n",
			"defaults: &store {" + field + ": GUARD_SECRET}\nstore: {<<: *store}\n",
		} {
			bindings, err := extractEnvironmentBindings([]byte(data))
			if err != nil {
				t.Fatal(err)
			}
			found := false
			for _, binding := range bindings {
				found = found || (binding.path == "/store/"+field && binding.name == "GUARD_SECRET")
			}
			if !found || ValidateEnvironmentBindings([]byte(data)) == nil {
				t.Fatalf("missed aliased binding for %s", field)
			}
		}
	}
}
