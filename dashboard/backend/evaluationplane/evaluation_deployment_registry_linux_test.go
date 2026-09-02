//go:build linux

package evaluationplane

import (
	"os"
	"path/filepath"
	"sync"
	"testing"
)

func TestDeploymentRegistryDescriptorPinsRootAcrossPathSubstitution(t *testing.T) {
	parent := t.TempDir()
	root := filepath.Join(parent, "deployments")
	if err := os.Mkdir(root, 0o700); err != nil {
		t.Fatal(err)
	}
	want := []byte(`{"schema_version":"evaluation-deployments.v1","deployments":[]}`)
	if err := os.WriteFile(filepath.Join(root, evaluationDeploymentRegistryFile), want, 0o600); err != nil {
		t.Fatal(err)
	}
	pinned, err := openDeploymentRegistryRoot(root)
	if err != nil {
		t.Fatal(err)
	}
	defer pinned.Close()
	moved := filepath.Join(parent, "pinned")
	if renameRootErr := os.Rename(root, moved); renameRootErr != nil {
		t.Fatal(renameRootErr)
	}
	attacker := filepath.Join(parent, "attacker")
	if createAttackerDirErr := os.Mkdir(attacker, 0o700); createAttackerDirErr != nil {
		t.Fatal(createAttackerDirErr)
	}
	if writeAttackerRegistryErr := os.WriteFile(filepath.Join(attacker, evaluationDeploymentRegistryFile), []byte(`{"secret":"outside"}`), 0o600); writeAttackerRegistryErr != nil {
		t.Fatal(writeAttackerRegistryErr)
	}
	if substituteRootErr := os.Symlink(attacker, root); substituteRootErr != nil {
		t.Fatal(substituteRootErr)
	}
	got, err := pinned.ReadFile(evaluationDeploymentRegistryFile, maxEvaluationDeploymentRegistrySize)
	if err != nil {
		t.Fatalf("read from pinned descriptor: %v", err)
	}
	if string(got) != string(want) {
		t.Fatalf("root substitution redirected descriptor read: got %q", got)
	}
	if _, err := openDeploymentRegistryRoot(root); err == nil {
		t.Fatal("a newly opened registry followed the substituted root symlink")
	}
}

func TestDeploymentRegistryNeverFollowsRacingConfigSymlink(t *testing.T) {
	root := t.TempDir()
	safeConfig := []byte(modelArmTestYAML)
	outsideConfig := []byte("version: v0.3\nrouting:\n  strategy: attacker\n  modelCards: []\n")
	writeDeploymentRegistryFixture(t, root, []evaluationDeploymentDefinition{{
		ID: "baseline", Name: "Baseline", ConfigFile: "config.yaml",
		RouterOrigin: "https://router.internal", EnvoyOrigin: "https://envoy.internal",
	}}, map[string][]byte{"config.yaml": safeConfig})
	outside := filepath.Join(t.TempDir(), "outside.yaml")
	if err := os.WriteFile(outside, outsideConfig, 0o600); err != nil {
		t.Fatal(err)
	}
	initial, err := LoadEvaluationDeploymentRegistry(root, "")
	if err != nil || len(initial) != 1 || initial[0].ConfigDigest != digestBytes(safeConfig) {
		t.Fatalf("safe registry precondition failed: targets=%+v err=%v", initial, err)
	}

	stop := make(chan struct{})
	var writer sync.WaitGroup
	writer.Add(1)
	go func() {
		defer writer.Done()
		for sequence := 0; ; sequence++ {
			select {
			case <-stop:
				return
			default:
			}
			temporary := filepath.Join(root, "replacement")
			_ = os.Remove(temporary)
			if sequence%2 == 0 {
				_ = os.Symlink(outside, temporary)
			} else {
				_ = os.WriteFile(temporary, safeConfig, 0o600)
			}
			_ = os.Rename(temporary, filepath.Join(root, "config.yaml"))
		}
	}()
	for range 200 {
		targets, err := LoadEvaluationDeploymentRegistry(root, "")
		if err != nil {
			continue
		}
		if len(targets) != 1 || targets[0].ConfigDigest != digestBytes(safeConfig) {
			close(stop)
			writer.Wait()
			t.Fatalf("racing symlink escaped registry root: %+v", targets)
		}
	}
	close(stop)
	writer.Wait()
}
