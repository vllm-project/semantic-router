//go:build !windows

package handlers

import (
	"bytes"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestOrdinaryRuntimeMutationRejectsAnyManagedMarkerWithoutMutation(t *testing.T) {
	for _, marker := range []string{"active.json", "activation-pending.json"} {
		t.Run(marker, func(t *testing.T) {
			root := t.TempDir()
			storeDir := filepath.Join(root, "recipe-store")
			if err := os.Mkdir(storeDir, 0o700); err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(filepath.Join(storeDir, marker), []byte("not even valid JSON"), 0o600); err != nil {
				t.Fatal(err)
			}
			t.Setenv(recipeStoreDirectoryEnv, storeDir)
			release, err := beginOrdinaryRuntimeConfigMutation(root)
			if release != nil {
				release()
			}
			var mutationErr *runtimeConfigMutationError
			if !errors.As(err, &mutationErr) || mutationErr.code != "managed_recipe_active" || mutationErr.status != http.StatusConflict {
				t.Fatalf("beginOrdinaryRuntimeConfigMutation() = %v", err)
			}
			if removeErr := os.Remove(filepath.Join(storeDir, marker)); removeErr != nil {
				t.Fatal(removeErr)
			}
			release, err = beginOrdinaryRuntimeConfigMutation(root)
			if err != nil {
				t.Fatalf("lock remained held: %v", err)
			}
			release()
		})
	}
}

func TestRuntimeMutationCrossProcessLockIsNonBlocking(t *testing.T) {
	storeDir := t.TempDir()
	first, exists, err := acquireRuntimeConfigStoreLock(storeDir)
	if err != nil || !exists {
		t.Fatalf("first lock = %#v, %v", first, err)
	}
	defer func() { _ = first.close() }()
	if second, _, err := acquireRuntimeConfigStoreLock(storeDir); !errors.Is(err, errRuntimeConfigLockBusy) {
		if second != nil {
			_ = second.close()
		}
		t.Fatalf("second lock error = %v", err)
	}
}

func TestRuntimeMutationCrossProcessLockWaitsForReplica(t *testing.T) {
	storeDir := t.TempDir()
	first, exists, err := acquireRuntimeConfigStoreLock(storeDir)
	if err != nil || !exists {
		t.Fatalf("first lock = %#v, %v", first, err)
	}
	released := make(chan struct{})
	go func() {
		time.Sleep(60 * time.Millisecond)
		_ = first.close()
		close(released)
	}()

	second, exists, err := acquireRuntimeConfigStoreLockWithin(storeDir, 500*time.Millisecond)
	if err != nil || !exists || second == nil {
		t.Fatalf("second lock = %#v, exists=%t, err=%v", second, exists, err)
	}
	_ = second.close()
	<-released
}

func TestRuntimeMutationLockPreservesSharedGroupAccess(t *testing.T) {
	storeDir := t.TempDir()
	lockPath := filepath.Join(storeDir, runtimeConfigLockName)
	if err := os.WriteFile(lockPath, nil, 0o660); err != nil {
		t.Fatal(err)
	}
	if err := os.Chmod(lockPath, 0o660); err != nil {
		t.Fatal(err)
	}

	lock, exists, err := acquireRuntimeConfigStoreLock(storeDir)
	if err != nil || !exists {
		t.Fatalf("lock = %#v, %v", lock, err)
	}
	if closeErr := lock.close(); closeErr != nil {
		t.Fatal(closeErr)
	}
	info, err := os.Stat(lockPath)
	if err != nil {
		t.Fatal(err)
	}
	if info.Mode().Perm() != 0o660 {
		t.Fatalf("lock mode = %o, want 660", info.Mode().Perm())
	}
}

func TestRuntimeMutationLockRejectsAccessForOtherUsers(t *testing.T) {
	storeDir := t.TempDir()
	lockPath := filepath.Join(storeDir, runtimeConfigLockName)
	if err := os.WriteFile(lockPath, nil, 0o606); err != nil {
		t.Fatal(err)
	}
	if err := os.Chmod(lockPath, 0o606); err != nil {
		t.Fatal(err)
	}

	lock, exists, err := acquireRuntimeConfigStoreLock(storeDir)
	if lock != nil {
		_ = lock.close()
	}
	if err == nil || exists {
		t.Fatalf("lock = %#v, exists = %t, err = %v", lock, exists, err)
	}
	if err.Error() != "runtime config lock must not grant access to other users" {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestUpdateConfigManagedMarkerReturnsStable409AndPreservesConfig(t *testing.T) {
	root := t.TempDir()
	configPath := createValidTestConfig(t, root)
	before, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatal(err)
	}
	storeDir := filepath.Join(root, "recipe-store")
	if mkdirErr := os.Mkdir(storeDir, 0o700); mkdirErr != nil {
		t.Fatal(mkdirErr)
	}
	if writeErr := os.WriteFile(filepath.Join(storeDir, "active.json"), []byte("{}"), 0o600); writeErr != nil {
		t.Fatal(writeErr)
	}
	t.Setenv(recipeStoreDirectoryEnv, storeDir)
	body, err := json.Marshal(canonicalConfigBody("127.0.0.1:8000"))
	if err != nil {
		t.Fatal(err)
	}
	response := httptest.NewRecorder()
	UpdateConfigHandler(configPath, false, root)(response, httptest.NewRequest(http.MethodPost, "/api/router/config/update", bytes.NewReader(body)))
	if response.Code != http.StatusConflict || !bytes.Contains(response.Body.Bytes(), []byte(`"error":"managed_recipe_active"`)) {
		t.Fatalf("response=%d body=%s", response.Code, response.Body.String())
	}
	after, err := os.ReadFile(configPath)
	if err != nil || !bytes.Equal(before, after) {
		t.Fatalf("config mutated: err=%v", err)
	}
}

func TestManagedMarkerGuardsAllOrdinaryRuntimeWriters(t *testing.T) {
	root := t.TempDir()
	configPath := createValidTestConfig(t, root)
	storeDir := filepath.Join(root, "recipe-store")
	if err := os.Mkdir(storeDir, 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(storeDir, "active.json"), []byte("{}"), 0o600); err != nil {
		t.Fatal(err)
	}
	t.Setenv(recipeStoreDirectoryEnv, storeDir)
	tests := []struct {
		name    string
		path    string
		body    string
		handler http.HandlerFunc
	}{
		{name: "router defaults", path: "/api/router/config/global/update", body: `{}`, handler: UpdateRouterDefaultsHandler(configPath, false, root)},
		{name: "global raw", path: "/api/router/config/global/raw/update", body: "router: {}\n", handler: UpdateGlobalConfigYAMLHandler(configPath, false, root)},
		{name: "deploy", path: "/api/router/config/deploy", body: `{"yaml":"routing: {}\\n"}`, handler: DeployHandler(configPath, false, root)},
		{name: "rollback", path: "/api/router/config/rollback", body: `{"version":"does-not-matter"}`, handler: RollbackHandler(configPath, false, root)},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			before, err := os.ReadFile(configPath)
			if err != nil {
				t.Fatal(err)
			}
			response := httptest.NewRecorder()
			test.handler(response, httptest.NewRequest(http.MethodPost, test.path, bytes.NewBufferString(test.body)))
			if response.Code != http.StatusConflict || !bytes.Contains(response.Body.Bytes(), []byte(`"error":"managed_recipe_active"`)) {
				t.Fatalf("response=%d body=%s", response.Code, response.Body.String())
			}
			after, err := os.ReadFile(configPath)
			if err != nil || !bytes.Equal(before, after) {
				t.Fatalf("config mutated: err=%v", err)
			}
		})
	}
}

func TestManagedMarkerGuardsSetupActivation(t *testing.T) {
	root := t.TempDir()
	storeDir := filepath.Join(root, "recipe-store")
	if err := os.Mkdir(storeDir, 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(storeDir, "active.json"), []byte("{}"), 0o600); err != nil {
		t.Fatal(err)
	}
	t.Setenv(recipeStoreDirectoryEnv, storeDir)

	setupPath := createBootstrapSetupConfig(t, root)
	setupBody, err := json.Marshal(SetupConfigRequest{Config: mustJSONRaw(t, createValidSetupPatch())})
	if err != nil {
		t.Fatal(err)
	}
	response := httptest.NewRecorder()
	SetupActivateHandler(setupPath, false, root)(response, httptest.NewRequest(http.MethodPost, "/api/setup/activate", bytes.NewReader(setupBody)))
	if response.Code != http.StatusConflict || !bytes.Contains(response.Body.Bytes(), []byte(`"error":"managed_recipe_active"`)) {
		t.Fatalf("setup response=%d body=%s", response.Code, response.Body.String())
	}
}

func TestRuntimeMutationRejectsSymlinkedStore(t *testing.T) {
	root := t.TempDir()
	realStore := filepath.Join(root, "real-store")
	if err := os.Mkdir(realStore, 0o700); err != nil {
		t.Fatal(err)
	}
	storeLink := filepath.Join(root, "store-link")
	if err := os.Symlink(realStore, storeLink); err != nil {
		t.Skipf("symlink unavailable: %v", err)
	}
	t.Setenv(recipeStoreDirectoryEnv, storeLink)
	_, err := beginOrdinaryRuntimeConfigMutation(root)
	var mutationErr *runtimeConfigMutationError
	if !errors.As(err, &mutationErr) || mutationErr.code != "config_coordination_failed" {
		t.Fatalf("symlinked store error = %v", err)
	}
}
