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

func TestResolveDynamicEnvoyTargetSkipsOffsetForStandaloneRouter(t *testing.T) {
	directory := t.TempDir()
	configPath := filepath.Join(directory, "runtime.yaml")
	if err := os.WriteFile(configPath, []byte("version: v0.3\nlisteners:\n  - name: public\n    address: 0.0.0.0\n    port: 9999\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	t.Setenv("VLLM_SR_GATEWAY", "standalone")
	t.Setenv("VLLM_SR_ENVOY_CONTAINER_NAME", "demo-envoy")
	t.Setenv("VLLM_SR_ROUTER_CONTAINER_NAME", "demo-router")
	t.Setenv("VLLM_SR_PORT_OFFSET", "100")
	target, err := resolveDynamicEnvoyTarget("http://demo-router:8899", configPath)
	if err != nil || target.String() != "http://demo-router:9999" {
		t.Fatalf("standalone router target picked up the host port offset: target=%v err=%v", target, err)
	}
}

func TestResolveDynamicEnvoyTargetSkipsOffsetForStackNamedStandaloneRouter(t *testing.T) {
	directory := t.TempDir()
	configPath := filepath.Join(directory, "runtime.yaml")
	if err := os.WriteFile(configPath, []byte("version: v0.3\nlisteners:\n  - name: public\n    address: 0.0.0.0\n    port: 9999\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	t.Setenv("VLLM_SR_GATEWAY", "standalone")
	t.Setenv("VLLM_SR_ENVOY_CONTAINER_NAME", "demo-envoy")
	t.Setenv("VLLM_SR_ROUTER_CONTAINER_NAME", "")
	t.Setenv("VLLM_SR_STACK_NAME", "demo")
	t.Setenv("VLLM_SR_PORT_OFFSET", "100")
	target, err := resolveDynamicEnvoyTarget("http://demo-vllm-sr-router-container:8899", configPath)
	if err != nil || target.String() != "http://demo-vllm-sr-router-container:9999" {
		t.Fatalf("stack-named standalone router target picked up the host port offset: target=%v err=%v", target, err)
	}
}

func TestResolveDynamicEnvoyTargetOffsetsRouterHostnameOutsideStandalone(t *testing.T) {
	directory := t.TempDir()
	configPath := filepath.Join(directory, "runtime.yaml")
	if err := os.WriteFile(configPath, []byte("version: v0.3\nlisteners:\n  - name: public\n    address: 0.0.0.0\n    port: 9999\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	t.Setenv("VLLM_SR_GATEWAY", "extproc")
	t.Setenv("VLLM_SR_ENVOY_CONTAINER_NAME", "demo-envoy")
	t.Setenv("VLLM_SR_ROUTER_CONTAINER_NAME", "demo-router")
	t.Setenv("VLLM_SR_PORT_OFFSET", "100")
	target, err := resolveDynamicEnvoyTarget("http://demo-router:8899", configPath)
	if err != nil || target.String() != "http://demo-router:10099" {
		t.Fatalf("router hostname skipped the offset outside standalone mode: target=%v err=%v", target, err)
	}
}
