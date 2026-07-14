package handlers

import (
	"bytes"
	"context"
	"errors"
	"os/exec"
	"sync"
	"time"
)

var errCommandOutputLimit = errors.New("external command output exceeded limit")

const containerCommandWaitDelay = 250 * time.Millisecond

type boundedCommandOutput struct {
	budget *boundedCommandOutputBudget
	buffer bytes.Buffer
}

type boundedCommandOutputBudget struct {
	mu        sync.Mutex
	remaining int
	exceeded  bool
}

type containerCommandOutput struct {
	stdout []byte
	stderr []byte
}

func (w *boundedCommandOutput) Write(data []byte) (int, error) {
	w.budget.mu.Lock()
	defer w.budget.mu.Unlock()

	written := len(data)
	remaining := w.budget.remaining
	if remaining > 0 {
		if remaining > len(data) {
			remaining = len(data)
		}
		_, _ = w.buffer.Write(data[:remaining])
		w.budget.remaining -= remaining
	}
	if remaining < len(data) {
		w.budget.exceeded = true
	}
	return written, nil
}

func (w *boundedCommandOutput) result() []byte {
	w.budget.mu.Lock()
	defer w.budget.mu.Unlock()

	return append([]byte(nil), w.buffer.Bytes()...)
}

func runBoundedCommand(
	ctx context.Context,
	runtimeBinary string,
	maxOutputBytes int,
	args ...string,
) ([]byte, error) {
	output, err := runBoundedCommandSplit(ctx, runtimeBinary, maxOutputBytes, args...)
	combined := make([]byte, 0, len(output.stdout)+len(output.stderr))
	combined = append(combined, output.stdout...)
	combined = append(combined, output.stderr...)
	return combined, err
}

func runBoundedCommandInDirectory(
	ctx context.Context,
	directory string,
	executable string,
	maxOutputBytes int,
	args ...string,
) ([]byte, error) {
	output, err := runBoundedCommandSplitInDirectory(
		ctx,
		directory,
		executable,
		maxOutputBytes,
		args...,
	)
	combined := make([]byte, 0, len(output.stdout)+len(output.stderr))
	combined = append(combined, output.stdout...)
	combined = append(combined, output.stderr...)
	return combined, err
}

func runBoundedCommandSplit(
	ctx context.Context,
	runtimeBinary string,
	maxOutputBytes int,
	args ...string,
) (containerCommandOutput, error) {
	return runBoundedCommandSplitInDirectory(ctx, "", runtimeBinary, maxOutputBytes, args...)
}

func runBoundedCommandSplitInDirectory(
	ctx context.Context,
	directory string,
	runtimeBinary string,
	maxOutputBytes int,
	args ...string,
) (containerCommandOutput, error) {
	budget := &boundedCommandOutputBudget{remaining: maxOutputBytes}
	stdout := &boundedCommandOutput{budget: budget}
	stderr := &boundedCommandOutput{budget: budget}
	cmd := exec.CommandContext(ctx, runtimeBinary, args...) // #nosec G204 -- callers constrain the runtime and argv.
	cmd.Dir = directory
	cmd.Stdout = stdout
	cmd.Stderr = stderr
	cmd.WaitDelay = containerCommandWaitDelay
	err := cmd.Run()

	result := containerCommandOutput{stdout: stdout.result(), stderr: stderr.result()}
	budget.mu.Lock()
	exceeded := budget.exceeded
	budget.mu.Unlock()
	if exceeded {
		return result, errCommandOutputLimit
	}
	return result, err
}
