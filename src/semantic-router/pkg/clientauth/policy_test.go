package clientauth

import (
	"fmt"
	"reflect"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func bearer(envs ...string) *config.EntrypointClientAuth {
	return &config.EntrypointClientAuth{Type: config.ClientAuthBearerAPIKey, APIKeyEnvs: envs}
}

func env(values map[string]string) func(string) (string, bool) {
	return func(name string) (string, bool) {
		value, ok := values[name]
		return value, ok
	}
}

func TestPolicyAdmitsOnlyTheKeysOfEachEntrypoint(t *testing.T) {
	policy, problems := Compile([]config.EntrypointMapping{
		{ModelNames: []string{"company/fast", "company/fast-v2"}, Recipe: "shared", ClientAuth: bearer("FAST_CURRENT", "FAST_NEXT")},
		{ModelNames: []string{"company/slow"}, Recipe: "shared", ClientAuth: bearer("SLOW_KEY")},
		{ModelNames: []string{"company/public"}, Recipe: "shared"},
		{API: config.SystemOneAPI, ModelNames: []string{"company/fast"}, Recipe: "native"},
	}, env(map[string]string{"FAST_CURRENT": "fast-1", "FAST_NEXT": "fast-2", "SLOW_KEY": "slow-1"}))
	if problems != nil {
		t.Fatal(problems)
	}
	for _, tc := range []struct {
		api   config.InferenceAPI
		model string
		key   string
		want  bool
	}{
		{config.ChatAPI, "company/fast", "fast-1", true},
		{"", "company/fast", "fast-2", true},
		{config.ChatAPI, "company/fast-v2", "fast-1", true},
		{config.ChatAPI, "company/fast", "", false},
		{config.ChatAPI, "company/fast", "slow-1", false},
		{config.ChatAPI, "company/fast", "fast-1 ", false},
		{config.ChatAPI, "company/slow", "slow-1", true},
		{config.ChatAPI, "company/slow", "fast-1", false},
		{config.ChatAPI, "company/public", "", true},
		{config.ChatAPI, "company/public", "placeholder", true},
		{config.SystemOneAPI, "company/fast", "", true},
		{config.ChatAPI, "backend/model", "", true},
	} {
		if got := policy.Authorize(tc.api, tc.model, tc.key); got != tc.want {
			t.Errorf("Authorize(%q, %q, %q) = %v, want %v", tc.api, tc.model, tc.key, got, tc.want)
		}
	}
	if !policy.Requires(config.ChatAPI, "company/fast") || policy.Requires(config.SystemOneAPI, "company/fast") || policy.Requires(config.ChatAPI, "company/public") {
		t.Fatal("Requires must follow each declaration within its namespace")
	}
	if !policy.AnyRequired() {
		t.Fatal("AnyRequired = false with protected entrypoints")
	}
}

func TestPublicPoliciesRequireNothing(t *testing.T) {
	public, problems := Compile([]config.EntrypointMapping{{ModelNames: []string{"company/public"}, Recipe: "shared"}}, env(nil))
	if problems != nil {
		t.Fatal(problems)
	}
	var none *Policy
	for _, policy := range []*Policy{public, none} {
		if policy.AnyRequired() || policy.Requires(config.ChatAPI, "company/public") || !policy.Authorize(config.ChatAPI, "company/public", "") {
			t.Fatalf("policy %v protects a public entrypoint", policy)
		}
	}
}

func TestCompileReportsEveryUnusableKeyWithoutItsValue(t *testing.T) {
	policy, problems := Compile([]config.EntrypointMapping{
		{ModelNames: []string{"company/fast"}, Recipe: "shared", ClientAuth: bearer("FAST_KEY", "FAST_UNSET")},
		{ModelNames: []string{"company/slow"}, Recipe: "shared", ClientAuth: bearer("SLOW_EMPTY", "SLOW_SPACED")},
		{ModelNames: []string{"company/basic"}, Recipe: "shared", ClientAuth: &config.EntrypointClientAuth{Type: "basic"}},
	}, env(map[string]string{"FAST_KEY": "fast-1", "SLOW_EMPTY": "", "SLOW_SPACED": "secret-value\n"}))
	want := []Problem{
		{Path: "entrypoints[0].client_auth.api_key_envs[1]", Message: "environment variable FAST_UNSET is not set"},
		{Path: "entrypoints[1].client_auth.api_key_envs[0]", Message: "environment variable SLOW_EMPTY is empty"},
		{Path: "entrypoints[1].client_auth.api_key_envs[1]", Message: "environment variable SLOW_SPACED must hold one Bearer token: visible ASCII characters, no spaces"},
		{Path: "entrypoints[2].client_auth.type", Message: `unsupported type "basic"`},
	}
	if policy != nil || !reflect.DeepEqual(problems, want) {
		t.Fatalf("Compile() = %v, %+v; want no policy and %+v", policy, problems, want)
	}
	for _, problem := range problems {
		if strings.Contains(problem.Message, "secret-value") || strings.Contains(problem.Message, "fast-1") {
			t.Fatalf("problem %q quotes a key", problem.Message)
		}
	}
}

func TestPolicyKeepsNoKey(t *testing.T) {
	policy, problems := Compile([]config.EntrypointMapping{
		{ModelNames: []string{"company/fast"}, Recipe: "shared", ClientAuth: bearer("FAST_KEY")},
	}, env(map[string]string{"FAST_KEY": "plain-key-value"}))
	if problems != nil {
		t.Fatal(problems)
	}
	for _, formatted := range []string{fmt.Sprintf("%v", policy), fmt.Sprintf("%+v", *policy), fmt.Sprintf("%#v", *policy)} {
		if strings.Contains(formatted, "plain-key-value") {
			t.Fatalf("formatted policy %q holds the key", formatted)
		}
	}
}
