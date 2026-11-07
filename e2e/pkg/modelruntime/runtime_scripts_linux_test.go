//go:build linux

package modelruntime

import (
	"encoding/json"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
)

// The in-container scripts find a runtime by its socket argument, so a
// stand-in process with a runtime's command line is enough to run them here.
// The scripts read /proc, so the test runs only on Linux.
func TestRuntimeScriptsFindTheProcessServingASocket(t *testing.T) {
	python, err := exec.LookPath("python3")
	if err != nil {
		t.Skip("python3 is not installed")
	}
	socket := filepath.Join(t.TempDir(), "cpu-0-0123456789ab.sock")
	standIn := exec.Command(python, "-c", "import time; time.sleep(60)", "vllm-srun", "serve", "--uds", socket, "--threads", "3")
	if err := standIn.Start(); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = standIn.Process.Kill(); _ = standIn.Wait() })

	run := func(script string) string {
		output, err := exec.Command(python, "-c", script, socket).Output()
		if err != nil {
			t.Fatalf("script failed: %v", err)
		}
		return strings.TrimSpace(string(output))
	}
	var argv []string
	if err := json.Unmarshal([]byte(run(runtimeArgs)), &argv); err != nil {
		t.Fatal(err)
	}
	if strings.Join(argv[len(argv)-5:], " ") != "serve --uds "+socket+" --threads 3" {
		t.Fatalf("command line %q", argv)
	}
	if pid := run(killRuntime); pid != strconv.Itoa(standIn.Process.Pid) {
		t.Fatalf("killed %s, want %d", pid, standIn.Process.Pid)
	}
	if err := standIn.Wait(); err == nil || !strings.Contains(err.Error(), "killed") {
		t.Fatalf("the stand-in must be killed, got %v", err)
	}
	if output, err := exec.Command(python, "-c", runtimeArgs, socket+".gone").CombinedOutput(); err == nil || !strings.Contains(string(output), "no runtime process serves") {
		t.Fatalf("a socket nothing serves must fail: %v %s", err, output)
	}
}
