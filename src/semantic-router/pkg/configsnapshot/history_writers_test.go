package configsnapshot

import (
	"context"
	"fmt"
	"path/filepath"
	"sort"
	"sync"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestEveryDocumentHasItsOwnHistoryDirectory(t *testing.T) {
	t.Setenv(HistoryDirEnv, "")
	t.Setenv(config.ConfigBaseDirEnv, "")
	dir := t.TempDir()
	workspace := filepath.Join(dir, ".vllm-sr", "config-backups")
	for path, want := range map[string]string{
		filepath.Join(dir, "config.yaml"):                          workspace,
		filepath.Join(dir, ".vllm-sr", "runtime-config.yaml"):      workspace,
		filepath.Join(dir, "router-a.yaml"):                        filepath.Join(workspace, "router-a.yaml"),
		filepath.Join(dir, "router-b.yaml"):                        filepath.Join(workspace, "router-b.yaml"),
		filepath.Join(dir, "nested", "config.yaml"):                filepath.Join(dir, "nested", ".vllm-sr", "config-backups"),
		filepath.Join(dir, "nested", "..", "router-a.yaml"):        filepath.Join(workspace, "router-a.yaml"),
		filepath.Join(dir, "config.yaml.d", "config.override.yml"): filepath.Join(dir, "config.yaml.d", ".vllm-sr", "config-backups", "config.override.yml"),
	} {
		if got := ResolvePersistence(path).HistoryDir; got != want {
			t.Errorf("history of %s = %s, want %s", path, got, want)
		}
	}
	replica := filepath.Join(dir, "replica-state")
	t.Setenv(HistoryDirEnv, replica)
	if got := HistoryDir(filepath.Join(dir, "config.yaml")); got != replica {
		t.Fatalf("configured history = %s, want %s", got, replica)
	}
}

// Two Routers whose documents sit in one directory used to share a history,
// and the versions they served crossed.
func TestDocumentsInOneDirectoryKeepSeparateHistories(t *testing.T) {
	t.Setenv(HistoryDirEnv, "")
	dir := t.TempDir()
	routers := map[string]*Manager{}
	for _, name := range []string{"router-a.yaml", "router-b.yaml"} {
		routers[name] = persistentManager(t, HistoryDir(filepath.Join(dir, name)), 10)
		if _, err := routers[name].Install(context.Background(), documentUpdate(SourceStartup, name+"-0")); err != nil {
			t.Fatal(err)
		}
	}
	for i := 1; i <= 3; i++ {
		for _, name := range []string{"router-a.yaml", "router-b.yaml"} {
			snapshot, err := routers[name].Apply(context.Background(), documentUpdate(SourceFile, fmt.Sprintf("%s-%d", name, i)))
			if err != nil {
				t.Fatal(err)
			}
			if snapshot.Version() != uint64(i+1) {
				t.Fatalf("%s activated v%d, want v%d: its versions crossed the other document's", name, snapshot.Version(), i+1)
			}
		}
	}
	for name, router := range routers {
		reloaded, err := NewHistory(10, NewDirStore(HistoryDir(filepath.Join(dir, name))))
		if err != nil {
			t.Fatal(err)
		}
		records := reloaded.List()
		if len(records) != 4 || string(records[0].Document) != name+"-3" {
			t.Fatalf("%s history = %+v, want its own 4 records", name, records)
		}
		if latest, _ := router.History().Latest(); latest.Version != 4 {
			t.Fatalf("%s latest = v%d", name, latest.Version)
		}
	}
}

// Two writers of one document, such as two Routers on the same file, take
// turns: every activation takes a version past every one either recorded,
// so no version names two activations.
func TestWritersOfOneDocumentNeverShareAVersion(t *testing.T) {
	dir := t.TempDir()
	writers := []*Manager{persistentManager(t, dir, 100), persistentManager(t, dir, 100)}
	for _, writer := range writers {
		snapshot, err := writer.Install(context.Background(), documentUpdate(SourceStartup, "start"))
		if err != nil || snapshot.Version() != 1 {
			t.Fatalf("startup on the recorded document = v%d, %v; want v1 for both", snapshot.Version(), err)
		}
	}
	var last uint64 = 1
	for i := 1; i <= 4; i++ {
		snapshot, err := writers[i%2].Apply(context.Background(), documentUpdate(SourceFile, fmt.Sprintf("turn-%d", i)))
		if err != nil {
			t.Fatal(err)
		}
		if snapshot.Version() != last+1 {
			t.Fatalf("turn %d activated v%d after v%d", i, snapshot.Version(), last)
		}
		last = snapshot.Version()
	}

	var wg sync.WaitGroup
	var mu sync.Mutex
	activated := map[uint64]string{}
	for w, writer := range writers {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := range 8 {
				document := fmt.Sprintf("writer-%d-%d", w, i)
				snapshot, err := writer.Apply(context.Background(), documentUpdate(SourceFile, document))
				if err != nil {
					t.Error(err)
					return
				}
				mu.Lock()
				if other, taken := activated[snapshot.Version()]; taken {
					t.Errorf("v%d activated %s and %s", snapshot.Version(), other, document)
				}
				activated[snapshot.Version()] = document
				mu.Unlock()
			}
		}()
	}
	wg.Wait()

	reloaded, err := NewHistory(100, NewDirStore(dir))
	if err != nil {
		t.Fatal(err)
	}
	records := reloaded.List()
	versions := make([]uint64, 0, len(records))
	for i := len(records) - 1; i >= 0; i-- {
		versions = append(versions, records[i].Version)
	}
	if len(versions) != 21 || !sort.SliceIsSorted(versions, func(i, j int) bool { return versions[i] < versions[j] }) {
		t.Fatalf("recorded versions in write order = %v, want 1 to 21", versions)
	}
	for i, version := range versions {
		if version != uint64(i+1) {
			t.Fatalf("recorded versions = %v, want each of 1 to 21 once", versions)
		}
	}
}
