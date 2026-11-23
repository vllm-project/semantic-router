package clientauth

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func mustCompile(t *testing.T, values map[string]string, entrypoints ...config.EntrypointMapping) *Policy {
	t.Helper()
	policy, problems := Compile(entrypoints, false, env(values))
	if problems != nil {
		t.Fatal(problems)
	}
	return policy
}

func TestAdmitTakesOnlyABearerKeyTheNameAccepts(t *testing.T) {
	policy := mustCompile(t, map[string]string{"FAST_KEY": "fast-1"},
		config.EntrypointMapping{ModelNames: []string{"company/fast"}, Recipe: "shared", ClientAuth: bearer("FAST_KEY")},
		config.EntrypointMapping{ModelNames: []string{"company/public"}, Recipe: "shared"},
	)
	for _, tc := range []struct {
		model         string
		authorization string
		want          Decision
	}{
		{"company/fast", "Bearer fast-1", Admitted},
		{"company/fast", "bearer  fast-1", Admitted},
		{" company/fast ", "Bearer fast-1", Admitted},
		{" company/fast ", "", Unauthorized},
		{"company/fast", "", Unauthorized},
		{"company/fast", "fast-1", Unauthorized},
		{"company/fast", "Basic fast-1", Unauthorized},
		{"company/fast", "Bearer fast-2", Unauthorized},
		{"company/public", "", Admitted},
		{"company/public", "Bearer placeholder", Admitted},
		{"backend/model", "", Admitted},
	} {
		if got := Admit(policy, policy, config.ChatAPI, tc.model, tc.authorization); got != tc.want {
			t.Errorf("Admit(%q, %q) = %v, want %v", tc.model, tc.authorization, got, tc.want)
		}
	}
}

// A request is admitted only by a policy the configuration active once its
// target is known agrees with, whichever configuration leased its stream.
func TestAdmitRefusesANameAReloadProtectsDifferently(t *testing.T) {
	keys := map[string]string{"FAST_KEY": "fast-1", "FAST_KEY_NEXT": "fast-2"}
	fast := func(auth *config.EntrypointClientAuth) config.EntrypointMapping {
		return config.EntrypointMapping{ModelNames: []string{"company/fast"}, Recipe: "shared", ClientAuth: auth}
	}
	public := mustCompile(t, keys, fast(nil))
	protected := mustCompile(t, keys, fast(bearer("FAST_KEY")))
	rotated := mustCompile(t, keys, fast(bearer("FAST_KEY_NEXT")))
	both := mustCompile(t, keys, fast(bearer("FAST_KEY", "FAST_KEY_NEXT")))
	reordered := mustCompile(t, keys, fast(bearer("FAST_KEY_NEXT", "FAST_KEY")))
	recompiled := mustCompile(t, keys, fast(bearer("FAST_KEY")),
		config.EntrypointMapping{ModelNames: []string{"company/other"}, Recipe: "shared", ClientAuth: bearer("FAST_KEY_NEXT")})
	var none *Policy

	for _, tc := range []struct {
		name           string
		served, active *Policy
		authorization  string
		want           Decision
	}{
		{"protected by a reload", public, protected, "", Superseded},
		{"protected by a reload, with its key", public, protected, "Bearer fast-1", Superseded},
		{"made public by a reload", protected, public, "Bearer fast-1", Superseded},
		{"keys rotated by a reload", protected, rotated, "Bearer fast-1", Superseded},
		{"keys reordered by a reload", both, reordered, "Bearer fast-2", Admitted},
		{"a key added by a reload", protected, both, "Bearer fast-1", Superseded},
		{"another name changed", protected, recompiled, "Bearer fast-1", Admitted},
		{"another name changed, without a key", protected, recompiled, "", Unauthorized},
		{"no policy on either side", none, public, "", Admitted},
	} {
		if got := Admit(tc.served, tc.active, config.ChatAPI, "company/fast", tc.authorization); got != tc.want {
			t.Errorf("%s: Admit = %v, want %v", tc.name, got, tc.want)
		}
	}
}

func TestBearerToken(t *testing.T) {
	for authorization, want := range map[string]string{
		"Bearer abc":   "abc",
		"BEARER abc":   "abc",
		" Bearer abc ": "abc",
		"Bearer":       "",
		"Bearer ":      "",
		"Basic abc":    "",
		"abc":          "",
		"":             "",
	} {
		if got := BearerToken(authorization); got != want {
			t.Errorf("BearerToken(%q) = %q, want %q", authorization, got, want)
		}
	}
}
