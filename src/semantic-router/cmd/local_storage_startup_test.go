//go:build !windows

package main

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"syscall"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerruntime"
)

// Model downloads go to a closed port, so a startup that reached them before
// the storage check would fail with a download error instead.
func TestRunRouterProcessRefusesUnwritableStorageBeforeModelDownloads(t *testing.T) {
	t.Setenv("HF_ENDPOINT", "http://127.0.0.1:1")
	t.Setenv("HF_HOME", t.TempDir())
	// A directory under a regular file is unwritable for every user, root too.
	file := filepath.Join(t.TempDir(), "file")
	if err := os.WriteFile(file, nil, 0o600); err != nil {
		t.Fatal(err)
	}
	dir := filepath.Join(file, "data")
	configPath := filepath.Join(t.TempDir(), "config.yaml")
	document := fmt.Sprintf(`version: v0.3
global:
  stores:
    vector_store:
      enabled: true
      backend_type: memory
      file_storage_dir: %s
`, dir)
	if err := os.WriteFile(configPath, []byte(document), 0o600); err != nil {
		t.Fatal(err)
	}

	err := runRouterProcess(context.Background(), runtimeOptions{configPath: configPath, port: freePort(t)})
	want := "check local storage: global.stores.vector_store.file_storage_dir: " + dir + " is not writable"
	if err == nil || !strings.HasPrefix(err.Error(), want) {
		t.Fatalf("runRouterProcess() error = %v, want prefix %q", err, want)
	}
}

// The storage pages quote the error the default directory gets on a read-only
// root filesystem.
func TestStorageDocsQuoteTheStartupError(t *testing.T) {
	readOnly := routerruntime.UnwritableDir{
		LocalStorageDir: config.LocalStorageDir{
			Setting: "global.stores.vector_store.file_storage_dir", Path: config.DefaultVectorStoreFileStorageDir,
		},
		Err: fmt.Errorf("/var/lib: %w", syscall.EROFS),
	}
	quote := "```text\ncheck local storage: " + readOnly.Error() + "\n```"
	for _, page := range []string{
		filepath.Join("website", "docs", "installation", "storage-overview.md"),
		filepath.Join("website", "i18n", "zh-Hans", "docusaurus-plugin-content-docs", "current", "installation", "storage-overview.md"),
	} {
		content, err := os.ReadFile(filepath.Join("..", "..", "..", page))
		if err != nil {
			t.Fatal(err)
		}
		if !strings.Contains(string(content), quote) {
			t.Errorf("%s does not quote the startup error:\n%s", page, quote)
		}
	}
}
