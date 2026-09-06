//go:build linux

package evaluationplane

import (
	"errors"
	"os"
	"os/exec"
	"syscall"
)

func configureWorkerProcessGroup(cmd *exec.Cmd) {
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
}

func terminateWorkerProcessGroup(cmd *exec.Cmd) error {
	if cmd.Process == nil {
		return os.ErrProcessDone
	}
	if err := syscall.Kill(-cmd.Process.Pid, syscall.SIGKILL); err != nil {
		if errors.Is(err, syscall.ESRCH) {
			return os.ErrProcessDone
		}
		return err
	}
	return nil
}
