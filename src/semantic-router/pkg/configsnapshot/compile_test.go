package configsnapshot

import (
	"encoding/json"
	"errors"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

const testDocument = `
version: v0.3
listeners:
  - name: public
    address: 0.0.0.0
    port: 8899
    timeout: 300s
    api_keys: [client-key-1]
providers:
  defaults:
    model: local-model
  models:
    - name: local-model
      provider_model_id: acme/local
      backend_refs:
        - name: a
          provider: vllm
          endpoint: 192.0.2.1:8000/v1
          weight: 2
        - name: b
          provider: vllm
          endpoint: 192.0.2.2:8000/v1
    - name: hosted-model
      provider_model_id: acme/hosted
      backend_refs:
        - provider: vllm
          endpoint: hosted.example:443/v1
          api_key_env: TEST_HOSTED_KEY
routing:
  modelCards:
    - name: local-model
    - name: hosted-model
  decisions:
    - name: default-decision
      rules:
        operator: AND
        conditions: []
      modelRefs:
        - model: local-model
recipes:
  - name: coding
    routing:
      decisions:
        - name: coding-decision
          rules:
            operator: AND
            conditions: []
          modelRefs:
            - model: hosted-model
entrypoints:
  - model_names: [coding-auto]
    recipe: coding
`

func parseTestConfig(t *testing.T, document string) *config.RouterConfig {
	t.Helper()
	cfg, err := config.ParseYAMLBytes([]byte(document))
	if err != nil {
		t.Fatalf("ParseYAMLBytes() error = %v", err)
	}
	return cfg
}

func compileTest(t *testing.T, cfg *config.RouterConfig) *Resources {
	t.Helper()
	resources, err := Compile(cfg)
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}
	return resources
}

func mustGet(t *testing.T, resources *Resources, kind Kind, name string) *Resource {
	t.Helper()
	resource, ok := resources.Get(Ref{Kind: kind, Name: name})
	if !ok {
		var names []string
		for _, r := range resources.List(kind) {
			names = append(names, r.Name)
		}
		t.Fatalf("no %s %q; have %v", kind, name, names)
	}
	return resource
}

func refNames(refs []Ref) []string {
	names := make([]string, 0, len(refs))
	for _, ref := range refs {
		names = append(names, ref.String())
	}
	return names
}

func TestCompileNamesEveryResourceAndItsReferences(t *testing.T) {
	t.Setenv("TEST_HOSTED_KEY", "hosted-secret-value")
	resources := compileTest(t, parseTestConfig(t, testDocument))

	listener := mustGet(t, resources, KindListener, "public")
	if got := listener.Spec.(ListenerSpec); got.Port != 8899 || got.Timeout != "300s" {
		t.Fatalf("listener spec = %+v", got)
	}
	if got := refNames(listener.Refs); len(got) != 1 || got[0] != "secret/listener/public" {
		t.Fatalf("listener refs = %v", got)
	}

	route := mustGet(t, resources, KindRoute, "coding-auto")
	if got := refNames(route.Refs); len(got) != 1 || got[0] != "program/coding" {
		t.Fatalf("route refs = %v", got)
	}
	if route.Path != "entrypoints[0]" {
		t.Fatalf("route path = %q", route.Path)
	}

	coding := mustGet(t, resources, KindProgram, "coding")
	if got := refNames(coding.Refs); len(got) != 1 || got[0] != "cluster/hosted-model" {
		t.Fatalf("coding program refs = %v", got)
	}
	defaultProgram := mustGet(t, resources, KindProgram, string(config.DefaultRecipeName))
	if defaultProgram.Path != "routing" {
		t.Fatalf("default program path = %q", defaultProgram.Path)
	}
	if got := defaultProgram.Spec.(ProgramSpec).Decisions; len(got) != 1 || got[0] != "default-decision" {
		t.Fatalf("default program decisions = %v", got)
	}

	local := mustGet(t, resources, KindCluster, "local-model")
	if got := refNames(local.Refs); len(got) != 2 {
		t.Fatalf("local cluster refs = %v, want its two endpoints", got)
	}
	for _, ref := range local.Refs {
		endpoint := mustGet(t, resources, ref.Kind, ref.Name).Spec.(EndpointSpec)
		if endpoint.Cluster != "local-model" || endpoint.Port != 8000 {
			t.Fatalf("endpoint %s spec = %+v", ref, endpoint)
		}
	}
	hosted := mustGet(t, resources, KindCluster, "hosted-model")
	if got := refNames(hosted.Refs); len(got) != 2 || got[1] != "secret/hosted-model" {
		t.Fatalf("hosted cluster refs = %v, want its endpoint and secret", got)
	}
	secret := mustGet(t, resources, KindSecret, "hosted-model").Spec.(SecretSpec)
	if len(secret.Env) != 1 || secret.Env[0] != "TEST_HOSTED_KEY" || secret.Inline != 0 {
		t.Fatalf("hosted secret spec = %+v", secret)
	}
}

func TestCompiledResourcesNeverCarryCredentialValues(t *testing.T) {
	t.Setenv("TEST_HOSTED_KEY", "hosted-secret-value")
	resources := compileTest(t, parseTestConfig(t, testDocument))
	for _, kind := range Kinds {
		encoded, err := json.Marshal(resources.List(kind))
		if err != nil {
			t.Fatal(err)
		}
		for _, value := range []string{"hosted-secret-value", "client-key-1"} {
			if strings.Contains(string(encoded), value) {
				t.Fatalf("%s resources expose a credential value: %s", kind, encoded)
			}
		}
	}
}

// Hashes are what incremental rebuilds compare, so each change must move the
// hash of the resource it belongs to and leave every other resource alone.
func TestResourceHashesMoveOnlyWithTheirOwnConfiguration(t *testing.T) {
	t.Setenv("TEST_HOSTED_KEY", "hosted-secret-value")
	base := compileTest(t, parseTestConfig(t, testDocument))

	for _, tc := range []struct {
		name    string
		edit    func(string) string
		env     string
		changed []string
	}{
		{
			name:    "endpoint address",
			edit:    func(doc string) string { return strings.Replace(doc, "192.0.2.2:8000", "192.0.2.3:8000", 1) },
			changed: []string{"endpoint/local-model/local-model_b"},
		},
		{
			name: "recipe decision",
			edit: func(doc string) string {
				return strings.Replace(doc, "name: coding-decision", "name: coding-decision-2", 1)
			},
			changed: []string{"program/coding"},
		},
		{
			name:    "credential value",
			edit:    func(doc string) string { return doc },
			env:     "rotated-secret-value",
			changed: []string{"secret/hosted-model"},
		},
		{
			name:    "listener api key",
			edit:    func(doc string) string { return strings.Replace(doc, "client-key-1", "client-key-2", 1) },
			changed: []string{"secret/listener/public"},
		},
		{
			name:    "listener timeout",
			edit:    func(doc string) string { return strings.Replace(doc, "timeout: 300s", "timeout: 60s", 1) },
			changed: []string{"listener/public"},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if tc.env != "" {
				t.Setenv("TEST_HOSTED_KEY", tc.env)
			}
			next := compileTest(t, parseTestConfig(t, tc.edit(testDocument)))
			var changed []string
			for _, kind := range Kinds {
				for _, resource := range next.List(kind) {
					previous, ok := base.Get(resource.Ref)
					if !ok || previous.Hash != resource.Hash {
						changed = append(changed, resource.Ref.String())
					}
				}
			}
			if strings.Join(changed, ",") != strings.Join(tc.changed, ",") {
				t.Fatalf("changed resources = %v, want %v", changed, tc.changed)
			}
		})
	}
}

func TestCompileRejectsDuplicateListenerNames(t *testing.T) {
	cfg := parseTestConfig(t, testDocument)
	cfg.Listeners = append(cfg.Listeners, config.Listener{Name: "public", Address: "127.0.0.1", Port: 9000})
	_, err := Compile(cfg)
	reasons := ReasonsOf(err)
	if len(reasons) != 1 || reasons[0].Code != CodeDuplicateName || reasons[0].Stage != StageCompile ||
		reasons[0].Path != "listeners[public]" {
		t.Fatalf("Compile() reasons = %+v (err %v)", reasons, err)
	}
}

func TestCompileReportsEveryUnresolvedReference(t *testing.T) {
	cfg := parseTestConfig(t, testDocument)
	cfg.Entrypoints = append(cfg.Entrypoints,
		config.EntrypointMapping{ModelNames: []string{"ghost-a"}, Recipe: "missing-a"},
		config.EntrypointMapping{ModelNames: []string{"ghost-b"}, Recipe: "missing-b"},
	)
	_, err := Compile(cfg)
	var rejection *Rejection
	if !errors.As(err, &rejection) || rejection.Stage != StageCompile {
		t.Fatalf("Compile() error = %v, want a compile rejection", err)
	}
	if len(rejection.Reasons) != 2 {
		t.Fatalf("reasons = %+v, want one per dangling reference", rejection.Reasons)
	}
	for _, reason := range rejection.Reasons {
		if reason.Code != CodeUnresolvedReference || !strings.HasPrefix(reason.Path, "entrypoints[") {
			t.Fatalf("reason = %+v", reason)
		}
	}
	if !strings.Contains(err.Error(), "route/ghost-a references program/missing-a") {
		t.Fatalf("error = %q", err)
	}
}

func TestCompileScopesRepeatedEndpointNamesByCluster(t *testing.T) {
	cfg := &config.RouterConfig{}
	cfg.ModelConfig = map[string]config.ModelParams{"a": {}, "b": {}}
	cfg.VLLMEndpoints = []config.VLLMEndpoint{
		{Name: "shared", Model: "a", Address: "a.example", Port: 80},
		{Name: "shared", Model: "b", Address: "b.example", Port: 80},
		{Name: "shared", Model: "b", Address: "c.example", Port: 80},
	}
	resources := compileTest(t, cfg)
	var names []string
	for _, endpoint := range resources.List(KindEndpoint) {
		names = append(names, endpoint.Name)
	}
	if got := strings.Join(names, ","); got != "a/shared,b/shared,b/shared#2" {
		t.Fatalf("endpoint names = %s", got)
	}
}

func TestCompileRejectsAnEmptyConfiguration(t *testing.T) {
	_, err := Compile(nil)
	if reasons := ReasonsOf(err); len(reasons) != 1 || reasons[0].Code != CodeInvalidDocument {
		t.Fatalf("Compile(nil) = %v", err)
	}
}
