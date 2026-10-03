//go:build linux

package modelservice

import (
	"bytes"
	"os/exec"
	"slices"
	"strconv"
	"strings"
	"testing"
)

func TestStartPinnedRestrictsOnlyTheChild(t *testing.T) {
	cpus := allowedCPUs()
	if len(cpus) < 2 {
		t.Skip("needs two CPUs")
	}
	cmd := exec.Command("grep", "Cpus_allowed_list", "/proc/self/status")
	var out bytes.Buffer
	cmd.Stdout = &out
	if err := startPinned(cmd, cpus[1:2]); err != nil {
		t.Fatal(err)
	}
	if err := cmd.Wait(); err != nil {
		t.Fatal(err)
	}
	if got := strings.TrimSpace(strings.TrimPrefix(out.String(), "Cpus_allowed_list:")); got != strconv.Itoa(cpus[1]) {
		t.Fatalf("child allowed cpus %q, want %d", got, cpus[1])
	}
	if after := allowedCPUs(); !slices.Equal(after, cpus) {
		t.Fatalf("the router's affinity changed: %v -> %v", cpus, after)
	}
}
