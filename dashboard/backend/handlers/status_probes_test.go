package handlers

import (
	"context"
	"errors"
	"os"
	"os/exec"
	"testing"
	"time"
)

func TestGetDockerContainerStatusWithProbeReturnsTrimmedStatus(t *testing.T) {
	status := getDockerContainerStatusWithProbe(
		"router",
		time.Second,
		func(context.Context, string) ([]byte, error) {
			return []byte("running\n"), nil
		},
	)

	if status != "running" {
		t.Fatalf("status = %q, want running", status)
	}
}

func TestGetDockerContainerStatusWithProbeReturnsNotFoundForMissingContainer(t *testing.T) {
	status := getDockerContainerStatusWithProbe(
		"missing",
		time.Second,
		func(context.Context, string) ([]byte, error) {
			return []byte("Error: No such object: missing"), errors.New("inspect failed")
		},
	)

	if status != "not found" {
		t.Fatalf("status = %q, want not found", status)
	}
}

func TestGetDockerContainerStatusWithProbeReturnsUnknownWithoutDockerCLI(t *testing.T) {
	status := getDockerContainerStatusWithProbe(
		"missing",
		time.Second,
		func(context.Context, string) ([]byte, error) {
			return nil, &exec.Error{Name: "docker", Err: errors.New("not found")}
		},
	)

	if status != "unknown" {
		t.Fatalf("status = %q, want unknown", status)
	}
}

func TestRunDockerStatusProbeIgnoresSuccessfulStderr(t *testing.T) {
	fakeDir := t.TempDir()
	fakeDocker := fakeDir + "/docker"
	if err := os.WriteFile(
		fakeDocker,
		[]byte("#!/bin/sh\nprintf 'warning from daemon\\n' >&2\nprintf 'running\\n'\n"),
		0o755,
	); err != nil {
		t.Fatalf("write fake Docker CLI: %v", err)
	}
	t.Setenv("PATH", fakeDir+string(os.PathListSeparator)+os.Getenv("PATH"))

	if status := getDockerContainerStatus("router"); status != "running" {
		t.Fatalf("status = %q, want running", status)
	}
}

func TestManagedContainerRunningOrAbsentRejectsPresentNonRunningStates(t *testing.T) {
	for _, status := range []string{"created", "exited", "paused", "restarting", "dead", "removing", "unknown"} {
		t.Run(status, func(t *testing.T) {
			running, err := managedContainerRunningOrAbsent(status, "Envoy")
			if running || err == nil {
				t.Fatalf("status %q returned running=%v error=%v", status, running, err)
			}
		})
	}

	running, err := managedContainerRunningOrAbsent("running", "Envoy")
	if !running || err != nil {
		t.Fatalf("running status returned running=%v error=%v", running, err)
	}
	running, err = managedContainerRunningOrAbsent("not found", "Envoy")
	if running || err != nil {
		t.Fatalf("not-found status returned running=%v error=%v", running, err)
	}
}

func TestGetDockerContainerStatusWithProbeReturnsUnknownOnRuntimeError(t *testing.T) {
	status := getDockerContainerStatusWithProbe(
		"router",
		time.Second,
		func(context.Context, string) ([]byte, error) {
			return []byte("Cannot connect to the Docker daemon"), errors.New("inspect failed")
		},
	)

	if status != "unknown" {
		t.Fatalf("status = %q, want unknown", status)
	}
}

func TestGetDockerContainerStatusWithProbeReturnsUnknownOnTimeout(t *testing.T) {
	started := time.Now()
	status := getDockerContainerStatusWithProbe(
		"router",
		20*time.Millisecond,
		func(ctx context.Context, _ string) ([]byte, error) {
			<-ctx.Done()
			return nil, ctx.Err()
		},
	)

	if status != "unknown" {
		t.Fatalf("status = %q, want unknown", status)
	}
	if elapsed := time.Since(started); elapsed > 500*time.Millisecond {
		t.Fatalf("timed probe returned after %s, want at most 500ms", elapsed)
	}
}

func TestGetDockerContainerStatusWithProbeTimeoutWinsOverOutput(t *testing.T) {
	status := getDockerContainerStatusWithProbe(
		"router",
		20*time.Millisecond,
		func(ctx context.Context, _ string) ([]byte, error) {
			<-ctx.Done()
			return []byte("running\n"), ctx.Err()
		},
	)

	if status != "unknown" {
		t.Fatalf("status = %q, want unknown", status)
	}
}

func TestGetDockerContainerStatusWithProbeReturnsUnknownOnEmptyOutput(t *testing.T) {
	status := getDockerContainerStatusWithProbe(
		"router",
		time.Second,
		func(context.Context, string) ([]byte, error) {
			return []byte(" \n"), nil
		},
	)

	if status != "unknown" {
		t.Fatalf("status = %q, want unknown", status)
	}
}
