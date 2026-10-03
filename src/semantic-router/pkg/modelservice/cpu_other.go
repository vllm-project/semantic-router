//go:build !linux

package modelservice

import (
	"os/exec"
	"runtime"
)

func allowedCPUs() []int { return sequentialCPUs(runtime.NumCPU()) }

// startPinned starts cmd; only Linux pins runtime processes to their CPUs.
func startPinned(cmd *exec.Cmd, _ []int) error { return cmd.Start() }
