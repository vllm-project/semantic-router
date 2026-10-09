package cluster

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"
)

// fakeKindTools puts stub `kind` and `kubectl` executables first on PATH and
// returns the file they append their calls to. failCreates is how many leading
// `kind create cluster` calls exit non-zero, and clusters is the output of
// `kind get clusters`.
func fakeKindTools(t *testing.T, failCreates int, failDeletes bool, clusters string) string {
	t.Helper()

	bin := t.TempDir()
	logPath := filepath.Join(t.TempDir(), "calls.log")
	if err := os.WriteFile(logPath, nil, 0o600); err != nil {
		t.Fatal(err)
	}

	writeStub(t, filepath.Join(bin, "kind"), `#!/usr/bin/env bash
echo "kind $*" >>"${FAKE_KIND_LOG}"
if [[ "$1" == "get" ]]; then
  echo "${FAKE_KIND_CLUSTERS}"
  exit 0
fi
if [[ "$1" == "create" ]]; then
  creates=$(grep -c '^kind create cluster' "${FAKE_KIND_LOG}")
  if ((creates <= FAKE_KIND_FAIL_CREATES)); then
    echo "ERROR: failed to create cluster: transient node bootstrap failure" >&2
    echo "first-create-failed" >>"${FAKE_KIND_LOG}"
    exit 1
  fi
fi
if [[ "$1" == "delete" && "${FAKE_KIND_FAIL_DELETES}" == "1" ]]; then
  echo "ERROR: failed to delete cluster" >&2
  exit 1
fi
exit 0
`)

	writeStub(t, filepath.Join(bin, "kubectl"), `#!/usr/bin/env bash
echo "kubectl $*" >>"${FAKE_KIND_LOG}"
exit 0
`)

	t.Setenv("FAKE_KIND_LOG", logPath)
	t.Setenv("FAKE_KIND_CLUSTERS", clusters)
	t.Setenv("FAKE_KIND_FAIL_CREATES", strconv.Itoa(failCreates))
	t.Setenv("FAKE_KIND_FAIL_DELETES", strconv.Itoa(boolToInt(failDeletes)))
	t.Setenv("E2E_KIND_STORAGE_DIR", t.TempDir())
	t.Setenv("E2E_KIND_MODELS_DIR", t.TempDir())
	t.Setenv("PATH", bin+string(os.PathListSeparator)+os.Getenv("PATH"))
	return logPath
}

func boolToInt(value bool) int {
	if value {
		return 1
	}
	return 0
}

func writeStub(t *testing.T, path string, body string) {
	t.Helper()
	if err := os.WriteFile(path, []byte(body), 0o700); err != nil {
		t.Fatal(err)
	}
}

func recordedCalls(t *testing.T, logPath string) []string {
	t.Helper()
	content, err := os.ReadFile(logPath)
	if err != nil {
		t.Fatal(err)
	}
	calls := []string{}
	for _, line := range strings.Split(strings.TrimSpace(string(content)), "\n") {
		if line != "" {
			calls = append(calls, line)
		}
	}
	return calls
}

// bootstrapCalls keeps only the cluster lifecycle calls, so assertions do not
// depend on generated config paths or on kubeconfig lookups that follow them.
func bootstrapCalls(calls []string) []string {
	bootstrap := []string{}
	for _, call := range calls {
		fields := strings.Fields(call)
		if len(fields) < 3 || fields[0] != "kind" {
			continue
		}
		switch subcommand := strings.Join(fields[1:3], " "); subcommand {
		case "get clusters", "create cluster", "delete cluster":
			bootstrap = append(bootstrap, subcommand)
		}
	}
	return bootstrap
}

func assertCalls(t *testing.T, logPath string, want []string) {
	t.Helper()
	got := bootstrapCalls(recordedCalls(t, logPath))
	if strings.Join(got, ", ") != strings.Join(want, ", ") {
		t.Fatalf("kind calls = %v, want %v", got, want)
	}
}

func newRetryingCluster(t *testing.T) *KindCluster {
	t.Helper()
	cluster := NewKindCluster("retry-test", false)
	cluster.bootstrapDelay = 0
	return cluster
}

func TestCreateRetriesOnceAfterDeletingTheFailedCluster(t *testing.T) {
	logPath := fakeKindTools(t, 1, false, "")

	if err := newRetryingCluster(t).Create(context.Background()); err != nil {
		t.Fatalf("Create returned error: %v", err)
	}

	assertCalls(t, logPath, []string{"get clusters", "create cluster", "delete cluster", "create cluster"})
	calls := strings.Join(recordedCalls(t, logPath), "\n")
	if strings.Count(calls, "kubectl wait --for=condition=Ready") != 1 {
		t.Fatalf("the retried cluster must wait for readiness exactly once, calls:\n%s", calls)
	}
}

func TestCreateReportsTheBootstrapFailureWhenTheRetryAlsoFails(t *testing.T) {
	logPath := fakeKindTools(t, 2, false, "")

	err := newRetryingCluster(t).Create(context.Background())
	if err == nil || !strings.Contains(err.Error(), "failed to create cluster") {
		t.Fatalf("expected the bootstrap failure to surface, got %v", err)
	}

	assertCalls(t, logPath, []string{"get clusters", "create cluster", "delete cluster", "create cluster"})
}

func TestCreateDoesNotDeleteAClusterThatBootstrappedOnTheFirstAttempt(t *testing.T) {
	logPath := fakeKindTools(t, 0, false, "")

	if err := newRetryingCluster(t).Create(context.Background()); err != nil {
		t.Fatalf("Create returned error: %v", err)
	}

	assertCalls(t, logPath, []string{"get clusters", "create cluster"})
}

func TestCreateReusesAnExistingClusterWithoutBootstrapping(t *testing.T) {
	logPath := fakeKindTools(t, 0, false, "retry-test")

	if err := newRetryingCluster(t).Create(context.Background()); err != nil {
		t.Fatalf("Create returned error: %v", err)
	}

	assertCalls(t, logPath, []string{"get clusters"})
}

func TestCreateStillRetriesWhenDeletingTheFailedClusterFails(t *testing.T) {
	logPath := fakeKindTools(t, 1, true, "")

	if err := newRetryingCluster(t).Create(context.Background()); err != nil {
		t.Fatalf("a failing delete must not stop the retry: %v", err)
	}

	assertCalls(t, logPath, []string{"get clusters", "create cluster", "delete cluster", "create cluster"})
}

func TestCreateStopsWaitingWhenTheContextIsCanceled(t *testing.T) {
	logPath := fakeKindTools(t, 1, false, "")

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	// Cancel once the first attempt has failed for certain, so the cancellation
	// is observed by the retry path rather than by the initial existence check.
	go func() {
		for deadline := time.Now().Add(4 * time.Second); time.Now().Before(deadline); time.Sleep(time.Millisecond) {
			if content, err := os.ReadFile(logPath); err == nil && strings.Contains(string(content), "first-create-failed") {
				cancel()
				return
			}
		}
	}()

	cluster := NewKindCluster("retry-test", false)
	cluster.bootstrapDelay = 10 * time.Second

	done := make(chan error, 1)
	go func() { done <- cluster.Create(ctx) }()

	select {
	case err := <-done:
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("expected context cancellation, got %v", err)
		}
	case <-time.After(8 * time.Second):
		t.Fatal("Create ignored the canceled context and kept waiting between attempts")
	}
}
