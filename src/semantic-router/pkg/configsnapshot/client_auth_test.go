package configsnapshot

import (
	"context"
	"encoding/json"
	"errors"
	"reflect"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func bearerAuth(envs ...string) *config.EntrypointClientAuth {
	return &config.EntrypointClientAuth{Type: config.ClientAuthBearerAPIKey, APIKeyEnvs: envs}
}

// protectedConfig is a minimal configuration whose company/fast entrypoint
// accepts the keys envs hold, next to the public company/public.
func protectedConfig(hash string, envs ...string) *config.RouterConfig {
	cfg := testConfig(hash)
	cfg.Entrypoints = []config.EntrypointMapping{
		{ModelNames: []string{"company/fast"}, Recipe: config.DefaultRecipeName, ClientAuth: bearerAuth(envs...)},
		{ModelNames: []string{"company/public"}, Recipe: config.DefaultRecipeName},
	}
	return cfg
}

func TestCompileRecordsWhereEntrypointKeysComeFrom(t *testing.T) {
	t.Setenv("TEST_HOSTED_KEY", "hosted-secret-value")
	t.Setenv("CODING_KEY_CURRENT", "coding-secret-current")
	t.Setenv("CODING_KEY_NEXT", "coding-secret-next")
	cfg := parseTestConfig(t, testDocument)
	cfg.Entrypoints[0].ClientAuth = bearerAuth("CODING_KEY_CURRENT", "CODING_KEY_NEXT")
	resources := compileTest(t, cfg)

	secret := mustGet(t, resources, KindSecret, "entrypoint/chat/coding-auto")
	if secret.Path != "entrypoints[0].client_auth" {
		t.Fatalf("secret path = %q", secret.Path)
	}
	if got := secret.Spec.(SecretSpec); !reflect.DeepEqual(got, SecretSpec{Env: []string{"CODING_KEY_CURRENT", "CODING_KEY_NEXT"}}) {
		t.Fatalf("secret spec = %+v", got)
	}
	route := mustGet(t, resources, KindRoute, "coding-auto")
	if got := refNames(route.Refs); !reflect.DeepEqual(got, []string{"program/coding", "secret/entrypoint/chat/coding-auto"}) {
		t.Fatalf("route refs = %v", got)
	}

	published, err := json.Marshal(resources.List(KindSecret))
	if err != nil {
		t.Fatal(err)
	}
	for _, value := range []string{"coding-secret-current", "coding-secret-next"} {
		if strings.Contains(string(published), value) {
			t.Fatalf("published secrets %s hold a key", published)
		}
	}
}

func TestRouteHashFollowsTheClientPolicy(t *testing.T) {
	t.Setenv("FAST_KEY", "fast-1")
	t.Setenv("FAST_KEY_NEXT", "fast-2")
	hash := func(cfg *config.RouterConfig) string {
		t.Helper()
		return mustGet(t, compileTest(t, cfg), KindRoute, "company/fast").Hash
	}
	public := testConfig("h")
	public.Entrypoints = []config.EntrypointMapping{{ModelNames: []string{"company/fast"}, Recipe: config.DefaultRecipeName}}
	protected := hash(protectedConfig("h", "FAST_KEY"))
	if hash(public) == protected {
		t.Fatal("protecting an entrypoint left its route unchanged")
	}
	if hash(protectedConfig("h", "FAST_KEY", "FAST_KEY_NEXT")) == protected {
		t.Fatal("rotating an entrypoint's keys left its route unchanged")
	}
	if hash(protectedConfig("h", "FAST_KEY")) != protected {
		t.Fatal("equal policies compiled to different routes")
	}
}

func TestCompileRejectsEntrypointKeysThatDoNotResolve(t *testing.T) {
	t.Setenv("FAST_KEY", "fast-1")
	t.Setenv("FAST_KEY_EMPTY", "")
	_, err := Compile(protectedConfig("h", "FAST_KEY", "FAST_KEY_UNSET", "FAST_KEY_EMPTY"))
	var rejection *Rejection
	if !errors.As(err, &rejection) || rejection.Stage != StageCompile {
		t.Fatalf("Compile() error = %v, want a compile rejection", err)
	}
	want := []Reason{
		{Stage: StageCompile, Code: CodeUnresolvedReference, Path: "entrypoints[0].client_auth.api_key_envs[1]", Message: "environment variable FAST_KEY_UNSET is not set"},
		{Stage: StageCompile, Code: CodeUnresolvedReference, Path: "entrypoints[0].client_auth.api_key_envs[2]", Message: "environment variable FAST_KEY_EMPTY is empty"},
	}
	if !reflect.DeepEqual(rejection.Reasons, want) {
		t.Fatalf("reasons = %+v, want %+v", rejection.Reasons, want)
	}
	if strings.Contains(err.Error(), "fast-1") {
		t.Fatalf("rejection %q quotes a key", err)
	}
}

func TestStartupFailsWhenAnEntrypointKeyIsMissing(t *testing.T) {
	m := NewManager(Options{Runtime: &fakeRuntime{}})
	_, err := m.Install(context.Background(), Update{Origin: Origin{Source: SourceStartup}, Config: protectedConfig("h1", "FAST_KEY_UNSET")})
	if !errors.As(err, new(*Rejection)) || m.Active() != nil {
		t.Fatalf("Install() error = %v, active = %v; want the Router not to start", err, m.Active())
	}
}

func TestEachSnapshotKeepsTheClientPolicyItWasCompiledWith(t *testing.T) {
	t.Setenv("FAST_KEY", "fast-1")
	t.Setenv("FAST_KEY_NEXT", "fast-2")
	m := NewManager(Options{Runtime: &fakeRuntime{}})
	first, err := m.Install(context.Background(), Update{Origin: Origin{Source: SourceStartup}, Config: protectedConfig("h1", "FAST_KEY")})
	if err != nil {
		t.Fatal(err)
	}
	accepts := func(snapshot *Snapshot, key string) bool {
		return snapshot.ClientAuth().Authorize(config.ChatAPI, "company/fast", key)
	}
	if !accepts(first, "fast-1") || accepts(first, "fast-2") || !first.ClientAuth().Authorize(config.ChatAPI, "company/public", "") {
		t.Fatal("the installed snapshot does not serve its own policy")
	}

	_, err = m.Apply(context.Background(), Update{Origin: Origin{Source: SourceFile}, Config: protectedConfig("h2", "FAST_KEY", "FAST_KEY_UNSET")})
	if !errors.As(err, new(*Rejection)) {
		t.Fatalf("Apply() with an unset key = %v, want a rejection", err)
	}
	if m.Active() != first || !accepts(m.Active(), "fast-1") {
		t.Fatal("a rejected reload replaced the active policy")
	}

	rotated, err := m.Apply(context.Background(), Update{Origin: Origin{Source: SourceFile}, Config: protectedConfig("h3", "FAST_KEY", "FAST_KEY_NEXT")})
	if err != nil {
		t.Fatal(err)
	}
	if m.Active() != rotated || !accepts(rotated, "fast-1") || !accepts(rotated, "fast-2") {
		t.Fatal("a policy-only reload did not activate the rotated keys")
	}
	// A request that leased the first snapshot keeps authorizing against it.
	if accepts(first, "fast-2") {
		t.Fatal("a reload changed the policy of the snapshot it replaced")
	}
}

// Secrets are named after their owner, so a provider model may take a name
// that spells another owner's secret.
func TestSecretsOfDifferentOwnersNeverCollide(t *testing.T) {
	t.Setenv("TEST_HOSTED_KEY", "hosted-secret-value")
	t.Setenv("CODING_KEY", "coding-secret-value")
	for _, provider := range []string{"entrypoint/chat/coding-auto", "listener/public"} {
		t.Run(provider, func(t *testing.T) {
			cfg := parseTestConfig(t, strings.ReplaceAll(testDocument, "hosted-model", provider))
			cfg.Entrypoints[0].ClientAuth = bearerAuth("CODING_KEY")
			resources := compileTest(t, cfg)
			for _, name := range []string{"cluster/" + provider, "listener/public", "entrypoint/chat/coding-auto"} {
				mustGet(t, resources, KindSecret, name)
			}
		})
	}
}
