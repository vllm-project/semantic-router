package configlifecycle

import "testing"

func TestMountsIncludeConfigPath(t *testing.T) {
	configPath := "/tmp/workspace/config.yaml"

	if !mountsIncludeConfigPath([]containerMount{
		{Source: configPath, Destination: managedContainerConfigDestination},
	}, configPath) {
		t.Fatal("expected config mount to match")
	}

	if mountsIncludeConfigPath([]containerMount{
		{Source: "/tmp/other/config.yaml", Destination: managedContainerConfigDestination},
	}, configPath) {
		t.Fatal("expected different mounted config path to be ignored")
	}

	if mountsIncludeConfigPath([]containerMount{
		{Source: configPath, Destination: "/app/.vllm-sr"},
	}, configPath) {
		t.Fatal("expected non-config destination to be ignored")
	}
}
