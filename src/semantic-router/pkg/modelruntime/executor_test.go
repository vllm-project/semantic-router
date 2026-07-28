package modelruntime

import (
	"context"
	"errors"
	"slices"
	"strings"
	"testing"
)

func TestExecuteRunsDependenciesAndIgnoresBestEffortFailure(t *testing.T) {
	executionOrder := make([]string, 0, 3)
	record := func(name string) {
		executionOrder = append(executionOrder, name)
	}

	summary, err := Execute(context.Background(), []Task{
		{
			Name: "download",
			Run: func(context.Context) error {
				record("download")
				return nil
			},
		},
		{
			Name:         "initialize",
			Dependencies: []string{"download"},
			Run: func(context.Context) error {
				record("initialize")
				return nil
			},
		},
		{
			Name:         "warmup",
			Dependencies: []string{"initialize"},
			BestEffort:   true,
			Run: func(context.Context) error {
				record("warmup")
				return errors.New("warmup failed")
			},
		},
	}, Options{MaxParallelism: 3})
	if err != nil {
		t.Fatalf("Execute() returned unexpected error: %v", err)
	}

	if !slices.Equal(executionOrder, []string{"download", "initialize", "warmup"}) {
		t.Fatalf("unexpected execution order: %v", executionOrder)
	}
	if status := summary.Results["download"].Status; status != TaskSucceeded {
		t.Fatalf("download status = %s, want %s", status, TaskSucceeded)
	}
	if status := summary.Results["initialize"].Status; status != TaskSucceeded {
		t.Fatalf("initialize status = %s, want %s", status, TaskSucceeded)
	}
	if status := summary.Results["warmup"].Status; status != TaskFailed {
		t.Fatalf("warmup status = %s, want %s", status, TaskFailed)
	}
}

func TestExecuteSkipsDependentsWhenBestEffortDependencyFails(t *testing.T) {
	summary, err := Execute(context.Background(), []Task{
		{
			Name: "seed",
			Run: func(context.Context) error {
				return nil
			},
		},
		{
			Name:         "optional-warmup",
			Dependencies: []string{"seed"},
			BestEffort:   true,
			Run: func(context.Context) error {
				return errors.New("warmup failed")
			},
		},
		{
			Name:         "post-warmup-check",
			Dependencies: []string{"optional-warmup"},
			Run: func(context.Context) error {
				t.Fatal("dependent task should have been skipped")
				return nil
			},
		},
	}, Options{MaxParallelism: 3})
	if err != nil {
		t.Fatalf("Execute() returned unexpected error: %v", err)
	}

	if status := summary.Results["optional-warmup"].Status; status != TaskFailed {
		t.Fatalf("optional-warmup status = %s, want %s", status, TaskFailed)
	}
	if status := summary.Results["post-warmup-check"].Status; status != TaskSkipped {
		t.Fatalf("post-warmup-check status = %s, want %s", status, TaskSkipped)
	}
}

func TestExecuteReturnsErrorForRequiredFailure(t *testing.T) {
	summary, err := Execute(context.Background(), []Task{
		{
			Name: "required-init",
			Run: func(context.Context) error {
				return errors.New("boom")
			},
		},
		{
			Name:         "downstream",
			Dependencies: []string{"required-init"},
			Run: func(context.Context) error {
				t.Fatal("downstream task should have been skipped")
				return nil
			},
		},
	}, Options{MaxParallelism: 2})
	if err == nil {
		t.Fatal("Execute() returned nil error for required task failure")
	}
	if status := summary.Results["required-init"].Status; status != TaskFailed {
		t.Fatalf("required-init status = %s, want %s", status, TaskFailed)
	}
	if status := summary.Results["downstream"].Status; status != TaskSkipped {
		t.Fatalf("downstream status = %s, want %s", status, TaskSkipped)
	}
}

// Regression for the goroutine-panic class first reported in
// https://github.com/vllm-project/semantic-router/issues/1843.
//
// startTask used to run task.Run on a bare goroutine, so a panicking
// initializer aborted the router process instead of failing the task. The
// tests below assert that a panic
//
//  1. is contained rather than propagated, even for a BestEffort task,
//  2. is reported as a normal task failure so dependents still resolve, and
//  3. still delivers an outcome, so Execute returns instead of blocking.
//
// Each test would abort or hang the test binary if the recovery regressed.

func TestExecuteRecoversFromBestEffortTaskPanic(t *testing.T) {
	summary, err := Execute(context.Background(), []Task{
		{
			Name:       "optional-init",
			BestEffort: true,
			Run: func(context.Context) error {
				panic("model init exploded")
			},
		},
	}, Options{MaxParallelism: 1})
	if err != nil {
		t.Fatalf("Execute() returned unexpected error for best-effort panic: %v", err)
	}
	if status := summary.Results["optional-init"].Status; status != TaskFailed {
		t.Fatalf("optional-init status = %s, want %s", status, TaskFailed)
	}
	if resultErr := summary.Results["optional-init"].Error; resultErr == nil {
		t.Fatal("optional-init error is nil, want the recovered panic")
	} else if !strings.Contains(resultErr.Error(), "model init exploded") {
		t.Fatalf("optional-init error = %v, want it to carry the panic value", resultErr)
	}
}

func TestExecuteReturnsErrorWhenRequiredTaskPanics(t *testing.T) {
	summary, err := Execute(context.Background(), []Task{
		{
			Name: "required-init",
			Run: func(context.Context) error {
				panic("model init exploded")
			},
		},
		{
			Name:         "downstream",
			Dependencies: []string{"required-init"},
			Run: func(context.Context) error {
				t.Fatal("downstream task should have been skipped")
				return nil
			},
		},
	}, Options{MaxParallelism: 2})
	if err == nil {
		t.Fatal("Execute() returned nil error for panicking required task")
	}
	if status := summary.Results["required-init"].Status; status != TaskFailed {
		t.Fatalf("required-init status = %s, want %s", status, TaskFailed)
	}
	if status := summary.Results["downstream"].Status; status != TaskSkipped {
		t.Fatalf("downstream status = %s, want %s", status, TaskSkipped)
	}
}
