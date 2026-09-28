// Package storagetest owns explicit opt-in and fail-closed dependency checks for storage tests.
package storagetest

import (
	"os"
	"strings"
)

type Test interface {
	Helper()
	Fatalf(string, ...any)
	Skipf(string, ...any)
}

// Require keeps ordinary unit runs independent of locally cached services.
// A mandatory invocation must explicitly enable every selected backend.
func Require(t Test, backend string) {
	t.Helper()
	flag := "SKIP_" + strings.ToUpper(backend) + "_TESTS"
	if os.Getenv(flag) == "false" {
		return
	}
	Unavailable(t, backend, "set "+flag+"=false to enable this storage integration")
}

// Unavailable never turns an unmet mandatory prerequisite into a passing skip.
func Unavailable(t Test, backend string, reason any) {
	t.Helper()
	if os.Getenv("VLLM_SR_REQUIRE_STORAGE_TESTS") == "1" {
		t.Fatalf("required %s storage integration unavailable: %v", backend, reason)
		return
	}
	t.Skipf("optional %s storage integration unavailable: %v", backend, reason)
}
