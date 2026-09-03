//go:build !windows && cgo

package apiserver

import (
	"bytes"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"

	"golang.org/x/sys/unix"
)

func TestManagedRecipeStateBlocksEveryRouterConfigWriter(t *testing.T) {
	for _, stateName := range []string{activeRecipePointerName, activationJournalName} {
		t.Run(stateName, func(t *testing.T) { assertManagedStateBlocksWriters(t, stateName) })
	}
}

type configWriterTestCase struct {
	name    string
	request *http.Request
	handle  func(http.ResponseWriter, *http.Request)
}

func assertManagedStateBlocksWriters(t *testing.T, stateName string) {
	t.Helper()
	storeDir := t.TempDir()
	t.Setenv(recipeStoreDirEnv, storeDir)
	// Content is intentionally malformed: presence alone must fail closed.
	if err := os.WriteFile(filepath.Join(storeDir, stateName), []byte("not-json\n"), 0o600); err != nil {
		t.Fatalf("write managed state: %v", err)
	}
	configPath := filepath.Join(t.TempDir(), "config.yaml")
	original := []byte("sentinel config\n")
	if err := os.WriteFile(configPath, original, 0o600); err != nil {
		t.Fatalf("write config: %v", err)
	}
	server := &ClassificationAPIServer{configPath: configPath}
	for _, testCase := range managedConfigWriterTestCases(server) {
		t.Run(testCase.name, func(t *testing.T) {
			assertManagedWriterBlocked(t, testCase, configPath, original)
		})
	}
}

func managedConfigWriterTestCases(server *ClassificationAPIServer) []configWriterTestCase {
	return []configWriterTestCase{
		{name: "router patch", request: httptest.NewRequest(http.MethodPatch, "/config/router", bytes.NewBufferString(`{}`)), handle: server.handleConfigPatch},
		{name: "router put", request: httptest.NewRequest(http.MethodPut, "/config/router", bytes.NewBufferString(`{}`)), handle: server.handleConfigPut},
		{name: "router rollback", request: httptest.NewRequest(http.MethodPost, "/config/router/rollback", bytes.NewBufferString(`{}`)), handle: server.handleConfigRollback},
		{name: "recipe put", request: requestWithPathValue(http.MethodPut, "/config/router/recipes/named", "name", "named"), handle: server.handlePutRecipe},
		{name: "recipe delete", request: requestWithPathValue(http.MethodDelete, "/config/router/recipes/named", "name", "named"), handle: server.handleDeleteRecipe},
		{name: "kb create", request: httptest.NewRequest(http.MethodPost, "/config/kbs", bytes.NewBufferString(`{}`)), handle: server.handleCreateKnowledgeBase},
		{name: "kb update", request: requestWithPathValue(http.MethodPut, "/config/kbs/named", "name", "named"), handle: server.handleUpdateKnowledgeBase},
		{name: "kb delete", request: requestWithPathValue(http.MethodDelete, "/config/kbs/named", "name", "named"), handle: server.handleDeleteKnowledgeBase},
	}
}

func assertManagedWriterBlocked(t *testing.T, testCase configWriterTestCase, configPath string, original []byte) {
	t.Helper()
	response := httptest.NewRecorder()
	testCase.handle(response, testCase.request)
	if response.Code != http.StatusConflict {
		t.Fatalf("status = %d, want 409: %s", response.Code, response.Body.String())
	}
	if code := parseErrorResponse(t, response.Body.Bytes()); code != "MANAGED_RECIPE_ACTIVE" {
		t.Fatalf("error code = %q, want MANAGED_RECIPE_ACTIVE", code)
	}
	persisted, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatalf("read config: %v", err)
	}
	if !bytes.Equal(persisted, original) {
		t.Fatalf("managed-state rejection changed config: %q", persisted)
	}
}

func TestRecipeStoreConfigLockRejectsCrossProcessContention(t *testing.T) {
	storeDir := t.TempDir()
	t.Setenv(recipeStoreDirEnv, storeDir)
	lockFile, active, err := acquireRecipeStoreConfigLock()
	if err != nil {
		t.Fatalf("acquire first lock: %v", err)
	}
	if active {
		t.Fatal("new store unexpectedly active")
	}
	defer func() {
		_ = unix.Flock(int(lockFile.Fd()), unix.LOCK_UN)
		_ = lockFile.Close()
	}()

	server := &ClassificationAPIServer{configPath: filepath.Join(t.TempDir(), "config.yaml")}
	rr := httptest.NewRecorder()
	server.handleConfigPut(
		rr,
		httptest.NewRequest(http.MethodPut, "/config/router", bytes.NewBufferString(`{}`)),
	)
	if rr.Code != http.StatusConflict {
		t.Fatalf("status = %d, want 409: %s", rr.Code, rr.Body.String())
	}
	if code := parseErrorResponse(t, rr.Body.Bytes()); code != "DEPLOY_IN_PROGRESS" {
		t.Fatalf("error code = %q, want DEPLOY_IN_PROGRESS", code)
	}
}

func TestRecipeStoreConfigLockLeavesUnmanagedSourceWritersEnabled(t *testing.T) {
	storeDir := t.TempDir()
	t.Setenv(recipeStoreDirEnv, storeDir)
	server := &ClassificationAPIServer{}

	guard, ok := server.acquireConfigMutationGuard(httptest.NewRecorder())
	if !ok {
		t.Fatal("unmanaged Recipe store unexpectedly blocked config mutation")
	}
	guard.Release()

	lockPath := filepath.Join(storeDir, recipeConfigLockName)
	info, err := os.Lstat(lockPath)
	if err != nil {
		t.Fatalf("stat shared lock: %v", err)
	}
	if !info.Mode().IsRegular() || info.Mode()&os.ModeSymlink != 0 {
		t.Fatalf("shared lock mode = %v, want real regular file", info.Mode())
	}
}

func TestLocalRuntimeWorkspaceRequiresRecipeStoreGuardPath(t *testing.T) {
	t.Setenv(recipeStoreDirEnv, "")
	stateRoot := t.TempDir()
	t.Setenv(runtimeStateRootDirEnv, stateRoot)
	t.Setenv(runtimeConfigPathEnv, filepath.Join(stateRoot, ".vllm-sr", "runtime-config.audit.yaml"))

	server := &ClassificationAPIServer{}
	rr := httptest.NewRecorder()
	if guard, ok := server.acquireConfigMutationGuard(rr); ok {
		guard.Release()
		t.Fatal("local runtime workspace unexpectedly accepted a missing Recipe store")
	}
	if rr.Code != http.StatusInternalServerError {
		t.Fatalf("status = %d, want 500: %s", rr.Code, rr.Body.String())
	}
	if code := parseErrorResponse(t, rr.Body.Bytes()); code != "RECIPE_STATE_GUARD_ERROR" {
		t.Fatalf("error code = %q, want RECIPE_STATE_GUARD_ERROR", code)
	}
}

func TestRecipeStoreConfigLockRejectsSymlink(t *testing.T) {
	storeDir := t.TempDir()
	t.Setenv(recipeStoreDirEnv, storeDir)
	outside := filepath.Join(t.TempDir(), "outside.lock")
	if err := os.WriteFile(outside, nil, 0o600); err != nil {
		t.Fatalf("write outside lock: %v", err)
	}
	if err := os.Symlink(outside, filepath.Join(storeDir, recipeConfigLockName)); err != nil {
		t.Fatalf("symlink lock: %v", err)
	}

	server := &ClassificationAPIServer{}
	rr := httptest.NewRecorder()
	if guard, ok := server.acquireConfigMutationGuard(rr); ok {
		guard.Release()
		t.Fatal("symlinked shared lock unexpectedly accepted")
	}
	if rr.Code != http.StatusInternalServerError {
		t.Fatalf("status = %d, want 500: %s", rr.Code, rr.Body.String())
	}
	if code := parseErrorResponse(t, rr.Body.Bytes()); code != "RECIPE_STATE_GUARD_ERROR" {
		t.Fatalf("error code = %q, want RECIPE_STATE_GUARD_ERROR", code)
	}
}

func requestWithPathValue(method, target, key, value string) *http.Request {
	request := httptest.NewRequest(method, target, bytes.NewBufferString(`{}`))
	request.SetPathValue(key, value)
	return request
}
