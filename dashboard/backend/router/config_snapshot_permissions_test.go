package router

import (
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"testing"

	"github.com/vllm-project/semantic-router/dashboard/backend/config"
)

func TestRegisterConfigRoutesRestrictsExistingSnapshots(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("POSIX permission bits")
	}
	root := t.TempDir()
	state := filepath.Join(root, ".vllm-sr")
	backups := filepath.Join(state, "config-backups")
	if err := os.MkdirAll(backups, 0o755); err != nil {
		t.Fatal(err)
	}
	snapshot := filepath.Join(backups, "config.20260101-000000.yaml")
	if err := os.WriteFile(snapshot, []byte("ordinary prior config"), 0o644); err != nil {
		t.Fatal(err)
	}
	registerConfigRoutes(http.NewServeMux(), &config.Config{ConfigDir: root, AbsConfigPath: filepath.Join(root, "config.yaml")})
	for path, want := range map[string]os.FileMode{state: 0o755, backups: 0o700, snapshot: 0o600} {
		info, err := os.Stat(path)
		if err != nil {
			t.Fatal(err)
		}
		if info.Mode().Perm() != want {
			t.Errorf("%s: mode %04o, want %04o", path, info.Mode().Perm(), want)
		}
	}
}
