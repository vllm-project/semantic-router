//go:build !windows

package modelservice

import (
	"os"
	"os/exec"
	"syscall"
)

// newProcessGroup starts the runtime in its own process group, so a stop
// reaches every process it started.
func newProcessGroup(cmd *exec.Cmd) {
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
}

// requestStop asks the runtime's process group to terminate.
func requestStop(process *os.Process) error {
	return syscall.Kill(-process.Pid, syscall.SIGTERM)
}

// forceKill ends the runtime's process group.
func forceKill(process *os.Process) error {
	return syscall.Kill(-process.Pid, syscall.SIGKILL)
}
