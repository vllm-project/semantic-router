package config

import (
	"strings"
	"testing"
)

type canonicalEntrypointRuleValidationErrorCase struct {
	name    string
	extra   string
	wantErr string
}

var canonicalEntrypointRuleValidationErrorCases = []canonicalEntrypointRuleValidationErrorCase{
	{
		name: "both recipe and rules set",
		extra: `
entrypoints:
  - model_names: ["vllm-sr/both"]
    recipe: privacy
    rules:
      - name: r
        recipe: privacy
`,
		wantErr: "set either recipe or rules, not both",
	},
	{
		name: "neither recipe nor rules set",
		extra: `
entrypoints:
  - model_names: ["vllm-sr/neither"]
`,
		wantErr: "recipe cannot be empty",
	},
	{
		name: "empty rules with no recipe",
		extra: `
entrypoints:
  - model_names: ["vllm-sr/empty-rules"]
    rules: []
`,
		wantErr: "recipe cannot be empty",
	},
	{
		name: "empty rule name",
		extra: `
entrypoints:
  - model_names: ["vllm-sr/bad-name"]
    rules:
      - name: ""
        recipe: privacy
`,
		wantErr: "name cannot be empty",
	},
	{
		name: "duplicate rule name",
		extra: `
entrypoints:
  - model_names: ["vllm-sr/dup-name"]
    rules:
      - name: r
        recipe: privacy
        matches:
          - headers: [{name: x-a, value: "1"}]
      - name: r
        recipe: default
        matches:
          - headers: [{name: x-b, value: "2"}]
`,
		wantErr: "duplicate rule name",
	},
	{
		name: "unknown recipe reference from a rule",
		extra: `
entrypoints:
  - model_names: ["vllm-sr/bad-recipe-ref"]
    rules:
      - name: r
        recipe: does-not-exist
`,
		wantErr: "unknown recipe",
	},
	{
		name: "duplicate case-equivalent header name within one match",
		extra: `
entrypoints:
  - model_names: ["vllm-sr/dup-header"]
    rules:
      - name: r
        recipe: privacy
        matches:
          - headers:
              - {name: X-Tenant, value: "A"}
              - {name: x-tenant, value: "A"}
`,
		wantErr: "duplicate header",
	},
	{
		name: "multiple catch-all rules",
		extra: `
entrypoints:
  - model_names: ["vllm-sr/two-catchalls"]
    rules:
      - name: r1
        recipe: privacy
      - name: r2
        recipe: default
`,
		wantErr: "multiple catch-all rules",
	},
	{
		name: "ambiguous incomparable rules",
		extra: `
entrypoints:
  - model_names: ["vllm-sr/ambiguous"]
    rules:
      - name: by-tenant
        recipe: privacy
        matches:
          - headers: [{name: x-authz-tenant-id, value: "A"}]
      - name: by-plan
        recipe: default
        matches:
          - headers: [{name: x-plan, value: "gold"}]
`,
		wantErr: "ambiguous",
	},
	{
		name: "path matcher missing leading slash",
		extra: `
entrypoints:
  - model_names: ["vllm-sr/bad-path"]
    rules:
      - name: r
        recipe: privacy
        matches:
          - path: {type: exact, value: "v1/chat/completions"}
`,
		wantErr: "must start with",
	},
	{
		name: "unsupported path matcher type",
		extra: `
entrypoints:
  - model_names: ["vllm-sr/bad-path-type"]
    rules:
      - name: r
        recipe: privacy
        matches:
          - path: {type: regex, value: "/v1/.*"}
`,
		wantErr: "path.type must be",
	},
	{
		name: "unsupported header matcher type",
		extra: `
entrypoints:
  - model_names: ["vllm-sr/bad-header-type"]
    rules:
      - name: r
        recipe: privacy
        matches:
          - headers: [{name: x-a, type: regex, value: "1"}]
`,
		wantErr: "headers[\"x-a\"].type must be",
	},
}

func TestCanonicalEntrypointRuleValidationErrors(t *testing.T) {
	for _, c := range canonicalEntrypointRuleValidationErrorCases {
		t.Run(c.name, func(t *testing.T) {
			_, err := ParseYAMLBytes([]byte(recipeTestBaseYAML + recipeTestPrivacyBlockYAML + c.extra))
			if err == nil {
				t.Fatalf("expected an error containing %q, got nil", c.wantErr)
			}
			if !strings.Contains(err.Error(), c.wantErr) {
				t.Fatalf("expected error containing %q, got: %v", c.wantErr, err)
			}
		})
	}
}

func TestCanonicalEntrypointRulesValidConfigLoads(t *testing.T) {
	extra := `
entrypoints:
  - model_names: ["vllm-sr/tenant-auto"]
    rules:
      - name: tenant-a-user-b
        matches:
          - path: {type: exact, value: "/v1/chat/completions"}
            headers:
              - {name: x-authz-tenant-id, type: exact, value: "A"}
              - {name: x-authz-user-id, value: "B"}
        recipe: privacy
      - name: tenant-a-default
        matches:
          - headers: [{name: x-authz-tenant-id, value: "A"}]
        recipe: default
`
	cfg, err := ParseYAMLBytes([]byte(recipeTestBaseYAML + recipeTestPrivacyBlockYAML + extra))
	if err != nil {
		t.Fatalf("unexpected parse error: %v", err)
	}

	entrypoint, ok := cfg.EntrypointByModelName("vllm-sr/tenant-auto")
	if !ok {
		t.Fatal("expected vllm-sr/tenant-auto to be a claimed entrypoint alias")
	}
	if len(entrypoint.Rules) != 2 {
		t.Fatalf("got %d rules, want 2", len(entrypoint.Rules))
	}
	if entrypoint.Recipe != "" {
		t.Errorf("a rules-based entrypoint must not also carry a legacy Recipe, got %q", entrypoint.Recipe)
	}
}
