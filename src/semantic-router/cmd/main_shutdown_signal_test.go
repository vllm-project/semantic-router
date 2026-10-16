//go:build !windows

package main

import (
	"bufio"
	"bytes"
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"syscall"
	"testing"
	"time"
)

func TestSIGTERMCompletesRuntimeRetirementAfterManagementDeadline(t *testing.T) {
	if output := os.Getenv("VSR_SHUTDOWN_TEST_OUTPUT"); output != "" {
		ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGTERM)
		defer stop()
		fmt.Println("ready")
		<-ctx.Done()
		shutdownCtx, cancel := context.WithTimeout(context.Background(), 500*time.Millisecond)
		defer cancel()
		err := shutdownRouterComponents(shutdownCtx,
			func(ctx context.Context) error { <-ctx.Done(); return ctx.Err() },
			func(ctx context.Context) error {
				if err := ctx.Err(); err != nil {
					return err
				}
				return os.WriteFile(output, []byte("retired\n"), 0o600)
			}, nil, func(context.Context) error { return nil })
		if !errors.Is(err, context.DeadlineExceeded) {
			t.Fatalf("lost management failure: %v", err)
		}
		return
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	output := filepath.Join(t.TempDir(), "retirement.txt")
	executable, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	// #nosec G204 -- Re-execute the current test binary with fixed test arguments.
	cmd := exec.CommandContext(ctx, executable, "-test.run=^TestSIGTERMCompletesRuntimeRetirementAfterManagementDeadline$")
	cmd.Env = append(os.Environ(), "VSR_SHUTDOWN_TEST_OUTPUT="+output)
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		t.Fatal(err)
	}
	if err = cmd.Start(); err != nil {
		t.Fatal(err)
	}
	defer func() {
		if cmd.ProcessState == nil {
			_ = cmd.Process.Kill()
			_ = cmd.Wait()
		}
	}()
	scanner := bufio.NewScanner(stdout)
	if !scanner.Scan() || scanner.Text() != "ready" {
		t.Fatalf("child not ready: %s", stderr.String())
	}
	if err = cmd.Process.Signal(syscall.SIGTERM); err != nil {
		t.Fatal(err)
	}
	for scanner.Scan() {
	} // Keep the child's test output pipe drained.
	if err = cmd.Wait(); err != nil {
		t.Fatalf("SIGTERM child failed: %v %s", err, stderr.String())
	}
	data, err := os.ReadFile(output)
	if err != nil || string(data) != "retired\n" {
		t.Fatalf("process exited without retiring runtime: %v", err)
	}
}
