package handlers

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"strings"
	"sync"
	"time"
)

// packagedCLI runs modules of the vllm-sr CLI installed in the same runtime
// image as the Dashboard, so Python-owned contracts are not copied into Go.
type packagedCLI struct {
	pythonPath string
}

func newPackagedCLI(pythonPath string) packagedCLI {
	pythonPath = strings.TrimSpace(pythonPath)
	if pythonPath == "" {
		pythonPath = "python3"
	}
	return packagedCLI{pythonPath: pythonPath}
}

// run executes `python -m module args...` with stdin and returns its standard
// output. When the module exits unsuccessfully, the error wraps
// *exec.ExitError and the output written so far is still returned.
func (cli packagedCLI) run(ctx context.Context, timeout time.Duration, stdin []byte, module string, args ...string) ([]byte, error) {
	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	workingDirectory, err := os.MkdirTemp("", "vllm-sr-cli-")
	if err != nil {
		return nil, errors.New("packaged CLI working directory is unavailable")
	}
	defer os.RemoveAll(workingDirectory)

	command := exec.CommandContext( //nolint:gosec // The Python path is configured at Dashboard startup and modules are fixed, never from an HTTP request.
		ctx,
		cli.pythonPath,
		append([]string{"-m", module}, args...)...,
	)
	// Execute in an empty directory so results cannot be contaminated by a
	// process working directory containing config.yaml.
	command.Dir = workingDirectory
	command.Env = append(os.Environ(), "PYTHONUNBUFFERED=1")
	command.Stdin = bytes.NewReader(stdin)

	// Spool output directly to disk. Catalog size grows with every model,
	// evaluation, and reasoning effort, so a fixed stdout buffer is not a valid
	// contract boundary.
	output, err := os.CreateTemp("", "vllm-sr-cli-output-*.json")
	if err != nil {
		return nil, errors.New("packaged CLI output is unavailable")
	}
	defer func() {
		_ = output.Close()
		_ = os.Remove(output.Name())
	}()
	command.Stdout = output
	// Module diagnostics are intentionally not exposed through the API. Do not
	// retain an unbounded stderr payload containing installation details.
	command.Stderr = io.Discard
	runErr := command.Run()
	if runErr != nil && errors.Is(ctx.Err(), context.DeadlineExceeded) {
		return nil, fmt.Errorf("%s timed out", module)
	}
	if closeErr := output.Close(); closeErr != nil {
		return nil, errors.New("packaged CLI output could not be finalized")
	}
	payload, err := os.ReadFile(output.Name())
	if err != nil {
		return nil, errors.New("packaged CLI output could not be read")
	}
	if runErr != nil {
		return payload, fmt.Errorf("%s failed: %w", module, runErr)
	}
	return payload, nil
}

// packagedDocument keeps the first document load succeeds with. Packaged
// assets cannot change while the Dashboard runs; failed loads are retried.
type packagedDocument struct {
	load  func(context.Context) ([]byte, error)
	mu    sync.RWMutex
	cache []byte
}

func (document *packagedDocument) get(ctx context.Context) ([]byte, error) {
	document.mu.RLock()
	if len(document.cache) > 0 {
		cached := append([]byte(nil), document.cache...)
		document.mu.RUnlock()
		return cached, nil
	}
	document.mu.RUnlock()

	loaded, err := document.load(ctx)
	if err != nil {
		return nil, err
	}

	document.mu.Lock()
	if len(document.cache) == 0 {
		document.cache = append([]byte(nil), loaded...)
	}
	cached := append([]byte(nil), document.cache...)
	document.mu.Unlock()
	return cached, nil
}
