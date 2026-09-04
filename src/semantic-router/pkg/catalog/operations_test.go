package catalog

import "testing"

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
