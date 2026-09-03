package recipe

import (
	"errors"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
)

func TestParsedSnapshotCacheReusesUnchangedExactFiveSnapshot(t *testing.T) {
	service := NewService(Options{Directory: writeManagedRecipe(t)})
	first, _, err := service.loadSnapshot()
	if err != nil {
		t.Fatalf("first loadSnapshot(): %v", err)
	}
	second, _, err := service.loadSnapshot()
	if err != nil {
		t.Fatalf("second loadSnapshot(): %v", err)
	}
	if first != second {
		t.Fatal("unchanged fixed files were reparsed instead of reusing the immutable snapshot")
	}
}

func TestParsedSnapshotCacheDoesNotExposeMutableDerivedState(t *testing.T) {
	service := NewService(Options{Directory: writeManagedRecipe(t)})
	mutateReturnedDescriptor(t, service)
	mutateReturnedProbeList(t, service)
	mutateReturnedProbeDetail(t, service)
	assertCachedDescriptorUnchanged(t, service)
	assertCachedProbeListUnchanged(t, service)
	assertCachedProbeDetailUnchanged(t, service)
}

func mutateReturnedDescriptor(t *testing.T, service *Service) {
	t.Helper()
	descriptor, err := service.Describe()
	if err != nil {
		t.Fatal(err)
	}
	descriptor.Metadata.Name = "mutated"
	descriptor.Metadata.Tags[0] = "mutated"
}

func mutateReturnedProbeList(t *testing.T, service *Service) {
	t.Helper()
	list, err := service.ListProbes(ListOptions{})
	if err != nil {
		t.Fatal(err)
	}
	list.Facets.Tags["smoke"] = 999
	list.Items[0].Tags[0] = "mutated"
	list.Items[0].Expected.Signals["keywords"][0] = "mutated"
}

func mutateReturnedProbeDetail(t *testing.T, service *Service) {
	t.Helper()
	detail, _, err := service.Probe("lane-a", "variant-a")
	if err != nil {
		t.Fatal(err)
	}
	function, ok := detail.Tools[0]["function"].(map[string]any)
	if !ok {
		t.Fatalf("tool function = %#v", detail.Tools[0]["function"])
	}
	function["name"] = "mutated"
	choice := detail.ToolChoice.(map[string]any)
	choiceFunction := choice["function"].(map[string]any)
	choiceFunction["name"] = "mutated"
}

func assertCachedDescriptorUnchanged(t *testing.T, service *Service) {
	t.Helper()
	descriptor, err := service.Describe()
	if err != nil || descriptor.Metadata.Name != "Test Recipe" || descriptor.Metadata.Tags[0] != "test" {
		t.Fatalf("cached metadata was mutated: %#v, %v", descriptor.Metadata, err)
	}
}

func assertCachedProbeListUnchanged(t *testing.T, service *Service) {
	t.Helper()
	list, err := service.ListProbes(ListOptions{})
	if err != nil || list.Facets.Tags["smoke"] != 1 || list.Items[0].Tags[0] != "smoke" ||
		list.Items[0].Expected.Signals["keywords"][0] != "signal-a" {
		t.Fatalf("cached list state was mutated: %#v, %v", list, err)
	}
}

func assertCachedProbeDetailUnchanged(t *testing.T, service *Service) {
	t.Helper()
	detail, _, err := service.Probe("lane-a", "variant-a")
	if err != nil {
		t.Fatal(err)
	}
	function, ok := detail.Tools[0]["function"].(map[string]any)
	if !ok || function["name"] != "lookup" {
		t.Fatalf("cached nested probe state was mutated: %#v", detail.Tools)
	}
	assertNamedToolChoice(t, detail.ToolChoice, "lookup")
}

func TestParsedSnapshotCacheInvalidatesAllDerivedStateTogether(t *testing.T) {
	directory := writeManagedRecipe(t)
	service := NewService(Options{Directory: directory})
	first, firstDescriptor, err := service.loadSnapshot()
	if err != nil {
		t.Fatalf("first loadSnapshot(): %v", err)
	}
	refreshManagedRecipeCacheFixture(t, directory)
	second, secondDescriptor, err := service.loadSnapshot()
	if err != nil {
		t.Fatalf("second loadSnapshot(): %v", err)
	}
	assertRefreshedRecipeSnapshot(t, first, second, firstDescriptor, secondDescriptor)
}

func refreshManagedRecipeCacheFixture(t *testing.T, directory string) {
	t.Helper()
	metadata, err := os.ReadFile(filepath.Join(directory, "metadata.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	writeFile(t, directory, "metadata.yaml", strings.Replace(string(metadata), "name: Test Recipe", "name: Refreshed Recipe", 1))
	probes, err := os.ReadFile(filepath.Join(directory, "probes.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	writeFile(t, directory, "probes.yaml", string(probes)+`      - id: cache-refresh
        query: Show the refreshed cache entry.
        tags: [cache-refresh]
`)
}

func assertRefreshedRecipeSnapshot(
	t *testing.T,
	first, second *snapshot,
	firstDescriptor, secondDescriptor Descriptor,
) {
	t.Helper()
	if first == second || first.recipeDigest == second.recipeDigest {
		t.Fatal("changed fixed files did not atomically replace the parsed snapshot and digest")
	}
	if secondDescriptor.Metadata == nil || secondDescriptor.Metadata.Name != "Refreshed Recipe" {
		t.Fatalf("metadata = %#v", secondDescriptor.Metadata)
	}
	if secondDescriptor.Counts.Probes != 3 || second.facets.Tags["cache-refresh"] != 1 {
		t.Fatalf("counts/facets = %#v / %#v", secondDescriptor.Counts, second.facets)
	}
	if _, found := second.byID[probeKey("lane-a", "cache-refresh")]; !found {
		t.Fatal("new probe index was not published with the refreshed snapshot")
	}
	if secondDescriptor.Digests.Recipe == firstDescriptor.Digests.Recipe {
		t.Fatal("descriptor retained the previous exact-five digest")
	}
}

func TestParsedSnapshotCacheSerializesConcurrentPublication(t *testing.T) {
	service := NewService(Options{Directory: writeManagedRecipe(t)})
	const workers = 32
	results := make(chan *snapshot, workers)
	errorsSeen := make(chan error, workers)
	var wait sync.WaitGroup
	for range workers {
		wait.Add(1)
		go func() {
			defer wait.Done()
			parsed, _, err := service.loadSnapshot()
			if err != nil {
				errorsSeen <- err
				return
			}
			results <- parsed
		}()
	}
	wait.Wait()
	close(results)
	close(errorsSeen)
	for err := range errorsSeen {
		t.Fatalf("concurrent loadSnapshot(): %v", err)
	}
	var published *snapshot
	for parsed := range results {
		if published == nil {
			published = parsed
			continue
		}
		if parsed != published {
			t.Fatal("concurrent readers observed more than one parsed snapshot for the same fingerprint")
		}
	}
}

func TestParsedSnapshotCacheFailsClosedAfterSymlinkReplacement(t *testing.T) {
	directory := writeManagedRecipe(t)
	service := NewService(Options{Directory: directory})
	if _, _, err := service.loadSnapshot(); err != nil {
		t.Fatalf("prime loadSnapshot(): %v", err)
	}
	readme := filepath.Join(directory, "README.md")
	if err := os.Remove(readme); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(filepath.Join(directory, "config.yaml"), readme); err != nil {
		t.Fatal(err)
	}
	descriptor, err := service.Describe()
	if !errors.Is(err, ErrInvalid) || descriptor.SourceHealth.Status != "invalid" {
		t.Fatalf("Describe() = %#v, %v; want invalid after symlink replacement", descriptor, err)
	}
	if cachedSnapshotForTest(service) != nil {
		t.Fatal("symlink replacement left the previous parsed snapshot cached")
	}
}

func TestParsedSnapshotCacheFailsClosedAfterSizeBoundViolation(t *testing.T) {
	directory := writeManagedRecipe(t)
	service := NewService(Options{Directory: directory})
	if _, _, err := service.loadSnapshot(); err != nil {
		t.Fatalf("prime loadSnapshot(): %v", err)
	}
	writeFile(t, directory, "metadata.yaml", strings.Repeat("x", int(maxMetadataBytes)+1))
	descriptor, err := service.Describe()
	if !errors.Is(err, ErrInvalid) || descriptor.SourceHealth.Status != "invalid" {
		t.Fatalf("Describe() = %#v, %v; want invalid after size violation", descriptor, err)
	}
	if cachedSnapshotForTest(service) != nil {
		t.Fatal("size-bound violation left the previous parsed snapshot cached")
	}
}

func TestParsedSnapshotCachePreservesStaleActionPrecondition(t *testing.T) {
	directory := writeManagedRecipe(t)
	service := NewService(Options{Directory: directory})
	first, _, err := service.loadSnapshot()
	if err != nil {
		t.Fatalf("prime loadSnapshot(): %v", err)
	}
	writeFile(t, directory, "README.md", "# Changed Recipe\n")
	if _, err := service.RunPlan("lane-a", "variant-a", first.recipeDigest); !errors.Is(err, ErrStale) {
		t.Fatalf("RunPlan() error = %v, want ErrStale after cache invalidation", err)
	}
}

func cachedSnapshotForTest(service *Service) *snapshot {
	service.snapshots.mu.Lock()
	defer service.snapshots.mu.Unlock()
	return service.snapshots.value
}
