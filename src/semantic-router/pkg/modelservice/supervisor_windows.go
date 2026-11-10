//go:build windows

package modelservice

import (
	"os"
	"os/exec"
)

// Windows has no process groups and no signal a parent can send a child, so
// the stop kills the runtime outright. A process the runtime started itself
// is not signalled and keeps running; the grace period in terminate rarely
// elapses because the kill takes effect at once.

func newProcessGroup(cmd *exec.Cmd) {}

func requestStop(process *os.Process) error {
	return process.Kill()
}

func forceKill(process *os.Process) error {
	return process.Kill()
}
