package config

import "testing"

func TestDefaultRouterAPIURLForEnvironment(t *testing.T) {
	t.Run("host", func(t *testing.T) {
		if got := defaultRouterAPIURLForEnvironment(false); got != "http://localhost:8080" {
			t.Fatalf("defaultRouterAPIURLForEnvironment(false) = %q", got)
		}
	})

	t.Run("container default", func(t *testing.T) {
		t.Setenv("VLLM_SR_ROUTER_CONTAINER_NAME", "")
		if got := defaultRouterAPIURLForEnvironment(true); got != "http://vllm-sr-router-container:8080" {
			t.Fatalf("defaultRouterAPIURLForEnvironment(true) = %q", got)
		}
	})

	t.Run("container override", func(t *testing.T) {
		t.Setenv("VLLM_SR_ROUTER_CONTAINER_NAME", "lane-a-router")
		if got := defaultRouterAPIURLForEnvironment(true); got != "http://lane-a-router:8080" {
			t.Fatalf("defaultRouterAPIURLForEnvironment(true) = %q", got)
		}
	})
}
