package catalog

import "testing"

func TestResolveOperationIncludesQueryPolicy(t *testing.T) {
	registry, err := BuiltIn()
	if err != nil {
		t.Fatal(err)
	}
	tests := []struct {
		name      string
		provider  string
		protocol  string
		operation string
		basePath  string
		want      ResolvedOperation
	}{
		{
			name: "OpenAI Responses", provider: "openai", protocol: "openai/responses@1",
			operation: "create", basePath: "/v1",
			want: ResolvedOperation{Path: "/v1/responses"},
		},
		{
			name: "custom API root", provider: "openai", protocol: "openai/chat-completions@1",
			operation: "create", basePath: "/v1beta/openai",
			want: ResolvedOperation{Path: "/v1beta/openai/chat/completions"},
		},
		{
			name: "Anthropic inventory", provider: "anthropic", operation: "list_models",
			want: ResolvedOperation{Path: "/v1/models"},
		},
		{
			name: "Azure default Chat", provider: "azure-openai", operation: "create",
			basePath: "/openai/deployments/astra-prod",
			want: ResolvedOperation{
				Path: "/openai/deployments/astra-prod/chat/completions", UseAPIVersionQuery: true,
			},
		},
		{
			name: "Azure Responses", provider: "azure-openai", protocol: "openai/responses@1",
			operation: "create", basePath: "/openai/deployments/astra-prod",
			want: ResolvedOperation{Path: "/openai/v1/responses"},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			got, err := registry.ResolveOperation(test.provider, test.protocol, test.operation, test.basePath)
			if err != nil || got != test.want {
				t.Fatalf("operation = %+v, err = %v, want %+v", got, err, test.want)
			}
		})
	}
}

func TestResolveOperationPathUsesProtocolAndProviderData(t *testing.T) {
	registry, err := BuiltIn()
	if err != nil {
		t.Fatal(err)
	}

	path, err := registry.ResolveOperationPath("openai", "openai/responses@1", "create", "/v1")
	if err != nil || path != "/v1/responses" {
		t.Fatalf("responses path = %q, err = %v", path, err)
	}
	path, err = registry.ResolveOperationPath("openai", "openai/chat-completions@1", "create", "/v1beta/openai")
	if err != nil || path != "/v1beta/openai/chat/completions" {
		t.Fatalf("custom API root path = %q, err = %v", path, err)
	}
	path, err = registry.ResolveOperationPath("anthropic", "", "list_models", "")
	if err != nil || path != "/v1/models" {
		t.Fatalf("model inventory path = %q, err = %v", path, err)
	}
	path, err = registry.ResolveOperationPath("azure-openai", "", "create", "/openai/deployments/example")
	if err != nil || path != "/openai/deployments/example/chat/completions" {
		t.Fatalf("provider override path = %q, err = %v", path, err)
	}
}

func TestResolveOperationPathRejectsUnknownContract(t *testing.T) {
	registry, err := BuiltIn()
	if err != nil {
		t.Fatal(err)
	}
	if _, err := registry.ResolveOperationPath("openai", "anthropic/messages@1", "create", ""); err == nil {
		t.Fatal("unsupported provider protocol was accepted")
	}
	if _, err := registry.ResolveOperationPath("openai", "", "delete_model", ""); err == nil {
		t.Fatal("unknown operation was accepted")
	}
	if _, err := registry.ResolveOperationPath("azure-openai", "", "list_models", ""); err == nil {
		t.Fatal("undeclared provider operation was accepted")
	}
}

func TestResolveProtocolOperationPathDoesNotApplyProviderOverrides(t *testing.T) {
	registry, err := BuiltIn()
	if err != nil {
		t.Fatal(err)
	}
	path, err := registry.ResolveProtocolOperationPath("openai/chat-completions@1", "create")
	if err != nil || path != "/v1/chat/completions" {
		t.Fatalf("protocol create path = %q, err = %v", path, err)
	}
}

func TestSnowflakeCortexResolveOperationPath(t *testing.T) {
	registry, err := BuiltIn()
	if err != nil {
		t.Fatal(err)
	}
	// The operator base URL must include /api/v2/cortex/v1: the resolver strips
	// the protocol /v1 prefix whenever a base path is configured, and Snowflake
	// documents both endpoints under /api/v2/cortex/v1.
	basePath := "/api/v2/cortex/v1"
	path, err := registry.ResolveOperationPath("snowflake-cortex", "openai/chat-completions@1", "create", basePath)
	if err != nil || path != "/api/v2/cortex/v1/chat/completions" {
		t.Fatalf("chat create path = %q, err = %v", path, err)
	}
	path, err = registry.ResolveOperationPath("snowflake-cortex", "anthropic/messages@1", "create", basePath)
	if err != nil || path != "/api/v2/cortex/v1/messages" {
		t.Fatalf("messages create path = %q, err = %v", path, err)
	}
}
