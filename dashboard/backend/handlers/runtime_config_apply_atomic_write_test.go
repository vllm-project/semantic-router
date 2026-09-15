package handlers

import (
	"errors"
	"os"
	"path/filepath"
	"testing"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	fakeclientset "k8s.io/client-go/kubernetes/fake"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/k8s"
)

func TestWriteConfigAtomicallySucceeds(t *testing.T) {
	dir := t.TempDir()
	configPath := filepath.Join(dir, "config.yaml")

	if err := writeConfigAtomically(configPath, []byte("routing: {}\n")); err != nil {
		t.Fatalf("writeConfigAtomically: %v", err)
	}

	got, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatalf("read configPath: %v", err)
	}
	if string(got) != "routing: {}\n" {
		t.Fatalf("configPath content = %q, want %q", got, "routing: {}\n")
	}
	if _, err := os.Stat(configPath + ".tmp"); !os.IsNotExist(err) {
		t.Fatalf("expected .tmp file to be removed after a successful write, stat err = %v", err)
	}
}

// TestWriteConfigAtomicallyRenameFailureLeavesExistingConfigUntouched pins the
// invariant a failed rename must never fall back to a non-atomic direct
// write: it forces the rename step to fail (standing in for the EBUSY/EROFS
// a K8s ConfigMap subPath mount returns) and asserts configPath keeps its
// original content rather than being overwritten by a partial or full
// direct write.
func TestWriteConfigAtomicallyRenameFailureLeavesExistingConfigUntouched(t *testing.T) {
	dir := t.TempDir()
	configPath := filepath.Join(dir, "config.yaml")
	original := []byte("routing: {original: true}\n")
	if err := os.WriteFile(configPath, original, 0o644); err != nil {
		t.Fatalf("seed configPath: %v", err)
	}

	renameErr := errors.New("simulated rename failure: device or resource busy")
	old := atomicRename
	atomicRename = func(string, string) error { return renameErr }
	t.Cleanup(func() { atomicRename = old })

	err := writeConfigAtomically(configPath, []byte("routing: {new: true}\n"))
	if !errors.Is(err, renameErr) {
		t.Fatalf("writeConfigAtomically error = %v, want %v", err, renameErr)
	}

	got, readErr := os.ReadFile(configPath)
	if readErr != nil {
		t.Fatalf("read configPath after failed write: %v", readErr)
	}
	if string(got) != string(original) {
		t.Fatalf("configPath was modified by a failed atomic write: got %q, want unchanged %q", got, original)
	}
	if _, statErr := os.Stat(configPath + ".tmp"); !os.IsNotExist(statErr) {
		t.Fatalf("expected .tmp file to be cleaned up after a failed rename, stat err = %v", statErr)
	}
}

func TestWriteConfigAtomicallyTempWriteFailureCleansUp(t *testing.T) {
	dir := t.TempDir()
	// A directory in place of configPath's parent forces the initial
	// temp-file open to fail (ENOTDIR/EISDIR on the ".tmp" path).
	badParent := filepath.Join(dir, "not-a-dir")
	if err := os.WriteFile(badParent, []byte("x"), 0o644); err != nil {
		t.Fatalf("seed badParent: %v", err)
	}
	configPath := filepath.Join(badParent, "config.yaml")

	if err := writeConfigAtomically(configPath, []byte("routing: {}\n")); err == nil {
		t.Fatal("expected writeConfigAtomically to fail when the temp file cannot be created")
	}
}

// TestWriteConfigAtomicallyRoutesThroughConfigMapWhenDeclared covers issue
// #3688: on a Kubernetes deployment that has declared a ConfigMap write
// target, writeConfigAtomically must write there via the Kubernetes API
// instead of attempting the local file, which is a read-only ConfigMap mount
// on every shipped manifest.
func TestWriteConfigAtomicallyRoutesThroughConfigMapWhenDeclared(t *testing.T) {
	dir := t.TempDir()
	configPath := filepath.Join(dir, "config.yaml")

	t.Setenv(k8s.ConfigMapNameEnv, "semantic-router-config")
	t.Setenv(k8s.ConfigMapNamespaceEnv, "vllm-semantic-router-system")

	cm := &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{Namespace: "vllm-semantic-router-system", Name: "semantic-router-config"},
		Data:       map[string]string{"config.yaml": "routing: {original: true}\n"},
	}
	clientset := fakeclientset.NewSimpleClientset(cm)
	restore := stubInClusterConfigMapWriter(t, k8s.NewConfigMapWriter(clientset))
	defer restore()

	if err := writeConfigAtomically(configPath, []byte("routing: {new: true}\n")); err != nil {
		t.Fatalf("writeConfigAtomically: %v", err)
	}

	if _, err := os.Stat(configPath); !os.IsNotExist(err) {
		t.Fatalf("local configPath was written despite a declared ConfigMap target, stat err = %v", err)
	}

	updated, err := clientset.CoreV1().ConfigMaps("vllm-semantic-router-system").Get(t.Context(), "semantic-router-config", metav1.GetOptions{})
	if err != nil {
		t.Fatalf("get ConfigMap: %v", err)
	}
	if updated.Data["config.yaml"] != "routing: {new: true}\n" {
		t.Fatalf("ConfigMap config.yaml = %q, want the new document", updated.Data["config.yaml"])
	}
}

// TestWriteConfigAtomicallyRefusesControllerOwnedConfigMap covers the other
// half of the fix: an Operator-managed ConfigMap gets regenerated on every
// reconcile, so writing to it directly would be silently reverted. The
// caller gets a clear error instead of a write that looks like it worked.
func TestWriteConfigAtomicallyRefusesControllerOwnedConfigMap(t *testing.T) {
	dir := t.TempDir()
	configPath := filepath.Join(dir, "config.yaml")

	t.Setenv(k8s.ConfigMapNameEnv, "operator-managed-config")
	t.Setenv(k8s.ConfigMapNamespaceEnv, "ns")

	isController := true
	cm := &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: "ns",
			Name:      "operator-managed-config",
			OwnerReferences: []metav1.OwnerReference{{
				APIVersion: "vllm.ai/v1alpha1",
				Kind:       "SemanticRouter",
				Name:       "prod",
				Controller: &isController,
			}},
		},
		Data: map[string]string{"config.yaml": "routing: {original: true}\n"},
	}
	clientset := fakeclientset.NewSimpleClientset(cm)
	restore := stubInClusterConfigMapWriter(t, k8s.NewConfigMapWriter(clientset))
	defer restore()

	err := writeConfigAtomically(configPath, []byte("routing: {new: true}\n"))
	if !errors.Is(err, k8s.ErrConfigMapControllerOwned) {
		t.Fatalf("writeConfigAtomically error = %v, want ErrConfigMapControllerOwned", err)
	}
}

// TestWriteConfigAtomicallyKeepsLocalFileWithoutADeclaredTarget pins the
// default: a deployment that has not set the ConfigMap env vars (every local
// CLI, VM, and plain Docker deployment today) keeps writing the local file
// exactly as before, with no Kubernetes client involved at all.
func TestWriteConfigAtomicallyKeepsLocalFileWithoutADeclaredTarget(t *testing.T) {
	dir := t.TempDir()
	configPath := filepath.Join(dir, "config.yaml")

	if err := writeConfigAtomically(configPath, []byte("routing: {}\n")); err != nil {
		t.Fatalf("writeConfigAtomically: %v", err)
	}
	got, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatalf("read configPath: %v", err)
	}
	if string(got) != "routing: {}\n" {
		t.Fatalf("configPath content = %q, want the written document", got)
	}
}

// stubInClusterConfigMapWriter forces resolvedConfigMapWriter's lazy init to
// return writer on its next call and restores the package's real singleton
// state afterward. Needed because that init caches its result process-wide:
// without a reset here, whichever test exercises the Kubernetes path first
// would permanently pin every later test in this package to its writer.
func stubInClusterConfigMapWriter(t *testing.T, writer *k8s.ConfigMapWriter) func() {
	t.Helper()
	configMapWriterMu.Lock()
	origFactory := newInClusterConfigMapWriter
	origResult := configMapWriterResult
	origErr := configMapWriterErr
	origResolved := configMapWriterResolved

	newInClusterConfigMapWriter = func() (*k8s.ConfigMapWriter, error) { return writer, nil }
	configMapWriterResult = nil
	configMapWriterErr = nil
	configMapWriterResolved = false
	configMapWriterMu.Unlock()

	return func() {
		configMapWriterMu.Lock()
		newInClusterConfigMapWriter = origFactory
		configMapWriterResult = origResult
		configMapWriterErr = origErr
		configMapWriterResolved = origResolved
		configMapWriterMu.Unlock()
	}
}
