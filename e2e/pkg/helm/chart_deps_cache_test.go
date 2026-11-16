package helm

import (
	"context"
	"os"
	"path/filepath"
	"testing"
)

func writeChart(t *testing.T, dir, lock string) {
	t.Helper()
	if err := os.WriteFile(filepath.Join(dir, "Chart.yaml"), []byte("name: c\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if lock != "" {
		if err := os.WriteFile(filepath.Join(dir, "Chart.lock"), []byte(lock), 0o644); err != nil {
			t.Fatal(err)
		}
	}
}

func TestChartDepsCacheKeyFollowsTheLock(t *testing.T) {
	unlocked := t.TempDir()
	writeChart(t, unlocked, "")
	if got := chartDepsCacheDir(unlocked); got != "" {
		t.Fatalf("a chart without Chart.lock got cache entry %q", got)
	}
	local := t.TempDir()
	writeChart(t, local, "repository: file://../sub\n")
	if got := chartDepsCacheDir(local); got != "" {
		t.Fatalf("a chart with a file:// dependency got cache entry %q", got)
	}
	a, b := t.TempDir(), t.TempDir()
	writeChart(t, a, "redis 1.0.0\n")
	writeChart(t, b, "redis 1.0.1\n")
	if chartDepsCacheDir(a) == chartDepsCacheDir(b) {
		t.Fatal("charts with different locks share a cache entry")
	}
}

// A cached build is copied into the chart without touching Helm, so a second
// profile in the same job never re-adds repositories or downloads charts.
func TestPrepareLocalChartReusesCachedDependencies(t *testing.T) {
	chart := t.TempDir()
	writeChart(t, chart, "e2e-cache-test "+t.Name()+"\n")
	cached := chartDepsCacheDir(chart)
	t.Cleanup(func() { _ = os.RemoveAll(cached) })
	if err := os.MkdirAll(cached, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(cached, "redis.tgz"), []byte("built"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Join(chart, "charts"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(chart, "charts", "stale.tgz"), []byte("old"), 0o644); err != nil {
		t.Fatal(err)
	}
	t.Setenv("PATH", t.TempDir()) // no helm binary: any Helm call fails

	prepared, cleanup, err := NewDeployer("", false).prepareLocalChartWithDeps(context.Background(), chart)
	if err != nil {
		t.Fatal(err)
	}
	defer cleanup()
	data, err := os.ReadFile(filepath.Join(prepared, "charts", "redis.tgz"))
	if err != nil || string(data) != "built" {
		t.Fatalf("cached dependency not copied: %q, %v", data, err)
	}
	if _, err := os.Stat(filepath.Join(prepared, "charts", "stale.tgz")); !os.IsNotExist(err) {
		t.Fatalf("an archive left in the source chart survived a cache hit: %v", err)
	}
}
