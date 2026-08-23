package handlers

import (
	"os"
	"testing"
)

func TestMain(m *testing.M) {
	// Handler tests exercise runtime coordination explicitly with per-test
	// fixtures. Do not let the developer host's active deployment select a
	// Docker-backed sync path for otherwise isolated config tests.
	for _, key := range []string{
		"VLLM_SR_RUNTIME_CONFIG_PATH",
		"VLLM_SR_PLATFORM",
		"DASHBOARD_PLATFORM",
		"VLLM_SR_ROUTER_CONTAINER_NAME",
		"VLLM_SR_ENVOY_CONTAINER_NAME",
		"VLLM_SR_DASHBOARD_CONTAINER_NAME",
	} {
		_ = os.Unsetenv(key)
	}
	isRunningInContainer = func() bool { return false }
	os.Exit(m.Run())
}
