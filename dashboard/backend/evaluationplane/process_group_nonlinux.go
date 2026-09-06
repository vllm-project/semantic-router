//go:build !linux

package evaluationplane

import (
	"os"
	"os/exec"
)

// The production Dashboard worker runs in the Linux image where the whole
// process group is isolated. Keep non-Linux development builds fail-safe by
// terminating the direct child rather than silently leaving cancellation off.
func configureWorkerProcessGroup(_ *exec.Cmd) {}

func terminateWorkerProcessGroup(cmd *exec.Cmd) error {
	if cmd.Process == nil {
		return os.ErrProcessDone
	}
	return cmd.Process.Kill()
}
