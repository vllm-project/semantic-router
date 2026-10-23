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

// A standalone stack's Playground dials the Router container inside the stack
// network, where the listener keeps its configured port; the offset applies
// only to host publication.
func TestResolveDynamicEnvoyTargetSkipsOffsetForManagedRouter(t *testing.T) {
	cases := []struct {
		name, stack, routerName, target, want string
	}{
		{"default stack", "", "", "http://vllm-sr-router-container:8899", "http://vllm-sr-router-container:9999"},
		{"named stack", "lane-a", "", "http://lane-a-vllm-sr-router-container:8899", "http://lane-a-vllm-sr-router-container:9999"},
		{"normalized stack", " lab a ", "", "http://lab-a-vllm-sr-router-container:8899", "http://lab-a-vllm-sr-router-container:9999"},
		{"explicit container name", "", "demo-router", "http://demo-router:8899", "http://demo-router:9999"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			configPath := filepath.Join(t.TempDir(), "runtime.yaml")
			if err := os.WriteFile(configPath, []byte("version: v0.3\nlisteners:\n  - name: public\n    address: 0.0.0.0\n    port: 9999\n"), 0o600); err != nil {
				t.Fatal(err)
			}
			t.Setenv("VLLM_SR_STACK_NAME", tc.stack)
			t.Setenv("VLLM_SR_ROUTER_CONTAINER_NAME", tc.routerName)
			t.Setenv("VLLM_SR_ENVOY_CONTAINER_NAME", "")
			t.Setenv("VLLM_SR_PORT_OFFSET", "100")
			target, err := resolveDynamicEnvoyTarget(tc.target, configPath)
			if err != nil || target.String() != tc.want {
				t.Fatalf("target=%v err=%v", target, err)
			}
		})
	}
}
