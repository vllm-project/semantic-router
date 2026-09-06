//go:build linux

package evaluationplane

import (
	"bufio"
	"errors"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"syscall"
	"testing"
	"time"
)

func TestWorkerProcessGroupTerminationIncludesDescendants(t *testing.T) {
	cmd := exec.Command("/bin/sh", "-c", "sleep 30 & child=$!; echo $child; wait")
	configureWorkerProcessGroup(cmd)
	stdout, pipeErr := cmd.StdoutPipe()
	if pipeErr != nil {
		t.Fatalf("StdoutPipe: %v", pipeErr)
	}
	if err := cmd.Start(); err != nil {
		t.Fatalf("Start: %v", err)
	}
	scanner := bufio.NewScanner(stdout)
	if !scanner.Scan() {
		_ = terminateWorkerProcessGroup(cmd)
		_ = cmd.Wait()
		t.Fatalf("read descendant pid: %v", scanner.Err())
	}
	childPID, parseErr := strconv.Atoi(strings.TrimSpace(scanner.Text()))
	if parseErr != nil {
		_ = terminateWorkerProcessGroup(cmd)
		_ = cmd.Wait()
		t.Fatalf("parse descendant pid: %v", parseErr)
	}
	if err := syscall.Kill(childPID, 0); err != nil {
		_ = terminateWorkerProcessGroup(cmd)
		_ = cmd.Wait()
		t.Fatalf("descendant was not running before cancellation: %v", err)
	}
	if err := terminateWorkerProcessGroup(cmd); err != nil {
		t.Fatalf("terminate process group: %v", err)
	}
	_ = cmd.Wait()

	deadline := time.Now().Add(2 * time.Second)
	for {
		err := syscall.Kill(childPID, 0)
		if errors.Is(err, syscall.ESRCH) || linuxProcessIsZombie(childPID) {
			break
		}
		if time.Now().After(deadline) {
			t.Fatalf("worker descendant %d survived process-group termination: %v", childPID, err)
		}
		time.Sleep(10 * time.Millisecond)
	}
}

func linuxProcessIsZombie(pid int) bool {
	data, err := os.ReadFile("/proc/" + strconv.Itoa(pid) + "/stat")
	if err != nil {
		return false
	}
	closing := strings.LastIndexByte(string(data), ')')
	return closing >= 0 && len(data) > closing+2 && data[closing+2] == 'Z'
}
