package router

import (
	"os"
	"path/filepath"
	"testing"
)

func TestResolveDynamicEnvoyTargetFollowsManagedListenerPort(t *testing.T) {
	directory := t.TempDir()
	configPath := filepath.Join(directory, "runtime.yaml")
	if err := os.WriteFile(configPath, []byte("version: v0.3\nlisteners:\n  - name: public\n    address: 0.0.0.0\n    port: 9999\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	t.Setenv("VLLM_SR_ENVOY_CONTAINER_NAME", "demo-envoy")
	target, err := resolveDynamicEnvoyTarget("http://demo-envoy:8899", configPath)
	if err != nil || target.String() != "http://demo-envoy:9999" {
		t.Fatalf("target=%v err=%v", target, err)
	}
}

func TestResolveDynamicEnvoyTargetAppliesHostPortOffset(t *testing.T) {
	directory := t.TempDir()
	configPath := filepath.Join(directory, "runtime.yaml")
	if err := os.WriteFile(configPath, []byte("version: v0.3\nlisteners:\n  - name: public\n    address: 0.0.0.0\n    port: 9999\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	t.Setenv("VLLM_SR_ENVOY_CONTAINER_NAME", "demo-envoy")
	t.Setenv("VLLM_SR_PORT_OFFSET", "100")
	target, err := resolveDynamicEnvoyTarget("http://127.0.0.1:8899", configPath)
	if err != nil || target.String() != "http://127.0.0.1:10099" {
		t.Fatalf("target=%v err=%v", target, err)
	}
}
