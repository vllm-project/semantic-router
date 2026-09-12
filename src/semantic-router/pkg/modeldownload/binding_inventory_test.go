package modeldownload

import (
	"os"
	"path/filepath"
	"slices"
	"testing"

	"google.golang.org/protobuf/encoding/protowire"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func deploymentConfig(provider, artifact string) *config.RouterConfig {
	cfg := &config.RouterConfig{MoMRegistry: map[string]string{"models/old": "test/old", artifact: "test/new"}}
	cfg.ModelDeployments = map[string]config.ModelDeployment{"new": {Provider: provider, Artifact: artifact, Revision: "abc123"}}
	cfg.CategoryModel.ModelID = "models/old"
	cfg.CategoryMappingPath = "labels.json"
	cfg.ModelBindings = map[string]config.ModelBinding{"domain_classifier": {Deployment: "new", Contract: "label_distribution.v1", Adapter: "mmbert32k"}}
	cfg.Decisions = []config.Decision{{Name: "route", Rules: config.RuleNode{Type: config.SignalTypeDomain, Name: "billing"}}}
	return cfg
}

func TestBoundArtifactReplacesDefaultAndPinsRevision(t *testing.T) {
	cfg := deploymentConfig("ort", "models/new")
	binding := cfg.ModelBindings["domain_classifier"]
	binding.Head = "onnx/classifier.onnx"
	binding.MappingPath = "models/new/labels.json"
	cfg.ModelBindings["domain_classifier"] = binding
	specs, err := BuildModelSpecs(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if len(specs) != 1 || specs[0].LocalPath != "models/new" || specs[0].Revision != "abc123" {
		t.Fatalf("specs=%#v", specs)
	}
	for _, file := range []string{"onnx/classifier.onnx", "labels.json", "config.json", "tokenizer.json"} {
		if !slices.Contains(specs[0].RequiredFiles, file) {
			t.Fatalf("missing %s: %#v", file, specs[0])
		}
	}
	if len(specs[0].ExcludePatterns) != 0 || !specs[0].CheckONNX {
		t.Fatalf("ORT graphs excluded: %#v", specs[0])
	}
	if cfg.CategoryModel.ModelID != "models/old" {
		t.Fatal("source configuration mutated")
	}
}

func TestUnregisteredLocalArtifactsDoNotRequireRegistry(t *testing.T) {
	cfg := deploymentConfig("candle", filepath.Join(t.TempDir(), "custom-model"))
	cfg.MoMRegistry = nil
	specs, err := BuildModelSpecs(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if len(specs) != 0 {
		t.Fatalf("local artifacts became downloads: %#v", specs)
	}
	if err := EnsureModelsForConfig(cfg); err != nil {
		t.Fatal(err)
	}
}

func TestSeparateBoundHeadAndMappingSnapshots(t *testing.T) {
	cfg := deploymentConfig("candle", "models/backbone")
	cfg.MoMRegistry["models/head"] = "test/head"
	cfg.MoMRegistry["models/mappings"] = "test/maps"
	binding := cfg.ModelBindings["domain_classifier"]
	binding.Head = "models/head"
	binding.MappingPath = "models/mappings/domain.json"
	cfg.ModelBindings["domain_classifier"] = binding
	specs, err := BuildModelSpecs(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if len(specs) != 3 {
		t.Fatalf("specs=%#v", specs)
	}
	mapping, ok := findSpecByPath(specs, "models/mappings")
	if !ok || !mapping.FilesOnly || !slices.Contains(mapping.RequiredFiles, "domain.json") {
		t.Fatalf("mapping=%#v", mapping)
	}
}

func TestSharedCandleORTArtifactKeepsBothFormats(t *testing.T) {
	inventory := modelInventory{registry: map[string]string{"models/shared": "test/shared"}, specs: map[string]ModelSpec{}}
	for _, provider := range []string{"candle", "ort"} {
		if err := inventory.addDeployment(&config.RouterConfig{}, config.ResolvedModelBinding{Deployment: config.ModelDeployment{Provider: provider, Artifact: "models/shared"}, Binding: config.ModelBinding{Contract: "label_distribution.v1"}}); err != nil {
			t.Fatal(err)
		}
	}
	spec := inventory.specs["models/shared"]
	if len(spec.ExcludePatterns) != 0 || len(spec.RequiredFileGroups) != 2 {
		t.Fatalf("spec=%#v", spec)
	}
}

func TestPinnedRevisionRefreshesAnAlreadyCompleteSnapshot(t *testing.T) {
	dir := t.TempDir()
	for _, name := range []string{"config.json", "model.safetensors"} {
		if err := os.WriteFile(filepath.Join(dir, name), []byte("test"), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	specs := []ModelSpec{{LocalPath: dir, RepoID: "test/model", Revision: "abc123"}}
	missing, err := GetMissingModels(specs)
	if err != nil {
		t.Fatal(err)
	}
	if len(missing) != 1 {
		t.Fatal("stale complete snapshot skipped revision resolution")
	}
	if args := buildDownloadArgs(missing[0]); !slices.Contains(args, "abc123") {
		t.Fatalf("args=%v", args)
	}
}

func protoBytes(field protowire.Number, data []byte) []byte {
	return protowire.AppendBytes(protowire.AppendTag(nil, field, protowire.BytesType), data)
}

func TestORTExternalTensorFilesAreRequiredWithoutInventingDataNames(t *testing.T) {
	dir := t.TempDir()
	graph := filepath.Join(dir, "model.onnx")
	entry := append(protoBytes(1, []byte("location")), protoBytes(2, []byte("actual-weights.bin"))...)
	model := protoBytes(7, protoBytes(5, protoBytes(13, entry)))
	if err := os.WriteFile(graph, model, 0o600); err != nil {
		t.Fatal(err)
	}
	present, err := onnxDependenciesPresent(graph)
	if err != nil || present {
		t.Fatalf("present=%v err=%v", present, err)
	}
	if writeErr := os.WriteFile(filepath.Join(dir, "actual-weights.bin"), []byte{1}, 0o600); writeErr != nil {
		t.Fatal(writeErr)
	}
	present, err = onnxDependenciesPresent(graph)
	if err != nil || !present {
		t.Fatalf("present=%v err=%v", present, err)
	}
	if writeErr := os.WriteFile(graph, protoBytes(7, protoBytes(5, protoBytes(9, []byte{1, 2, 3}))), 0o600); writeErr != nil {
		t.Fatal(writeErr)
	}
	present, err = onnxDependenciesPresent(graph)
	if err != nil || !present {
		t.Fatalf("inline tensors wrongly require .data: present=%v err=%v", present, err)
	}
}

func TestBindingsProvisionOnlyReachableRecipeArtifacts(t *testing.T) {
	cfg := deploymentConfig("candle", "models/default")
	cfg.ModelDeployments["private"] = config.ModelDeployment{Provider: "ort", Artifact: "models/private"}
	cfg.ModelDeployments["dormant"] = config.ModelDeployment{Provider: "candle", Artifact: "models/dormant"}
	cfg.MoMRegistry["models/private"] = "test/private"
	cfg.MoMRegistry["models/dormant"] = "test/dormant"
	profile := func(deployment string) config.RoutingProfile {
		return config.RoutingProfile{ModelBindings: map[string]config.ModelBinding{"domain_classifier": {Deployment: deployment, Contract: "label_distribution.v1", Adapter: "mmbert32k"}}, Decisions: cfg.Decisions}
	}
	cfg.Recipes = []config.RoutingRecipe{{Name: config.DefaultRecipeName, Profile: profile("new")}, {Name: "private", Profile: profile("private")}, {Name: "dormant", Profile: profile("dormant")}}
	cfg.Entrypoints = []config.EntrypointMapping{{ModelNames: []string{"private"}, Recipe: "private"}}
	specs, err := BuildModelSpecs(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if len(specs) != 2 {
		t.Fatalf("specs=%#v", specs)
	}
	for _, path := range []string{"models/default", "models/private"} {
		if _, ok := findSpecByPath(specs, path); !ok {
			t.Fatalf("missing %s", path)
		}
	}
}

func TestUnreachableDefaultStillProvisionsDeclaredPublicAPIModel(t *testing.T) {
	cfg := &config.RouterConfig{MoMRegistry: map[string]string{"models/fact": "test/fact"}}
	cfg.AutoModelNames = []string{}
	cfg.HallucinationMitigation.FactCheckModel.ModelID = "models/fact"
	cfg.FactCheckRules = []config.FactCheckRule{{Name: "api-check"}}
	specs, err := BuildModelSpecs(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if len(specs) != 1 || specs[0].LocalPath != "models/fact" {
		t.Fatalf("public API model omitted: %#v", specs)
	}
}

func TestConflictingPinnedRevisionsCannotShareDownloadDirectory(t *testing.T) {
	inventory := modelInventory{registry: map[string]string{"models/shared": "test/shared"}, specs: map[string]ModelSpec{}}
	if err := inventory.add(ModelSpec{LocalPath: "models/shared", Revision: "first"}); err != nil {
		t.Fatal(err)
	}
	if err := inventory.add(ModelSpec{LocalPath: "models/shared", Revision: "second"}); err == nil {
		t.Fatal("conflicting revisions accepted")
	}
}

func TestExplicitDeploymentDownloadFailsClosed(t *testing.T) {
	for _, exit := range []string{"0", "1"} {
		t.Run("exit_"+exit, func(t *testing.T) {
			command := filepath.Join(t.TempDir(), "hf")
			if err := os.WriteFile(command, []byte("#!/bin/sh\nexit "+exit+"\n"), 0o700); err != nil {
				t.Fatal(err)
			}
			previous := hfCommand
			hfCommand = command
			t.Cleanup(func() { hfCommand = previous })
			err := DownloadModel(ModelSpec{LocalPath: filepath.Join(t.TempDir(), "missing"), RepoID: "test/model", Revision: "pin", Strict: true}, DownloadConfig{})
			if err == nil {
				t.Fatal("explicit artifact accepted a failed or incomplete download")
			}
		})
	}
}

func writeHFSnapshot(t *testing.T, dir, revision string, extraFiles ...string) {
	t.Helper()
	for _, name := range append([]string{"config.json", "tokenizer.json", "model.safetensors"}, extraFiles...) {
		if err := os.WriteFile(filepath.Join(dir, name), []byte("fixture"), 0o600); err != nil {
			t.Fatal(err)
		}
		metadata := filepath.Join(dir, ".cache", "huggingface", "download", name+".metadata")
		if err := os.MkdirAll(filepath.Dir(metadata), 0o700); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(metadata, []byte(revision+"\nfixture-etag\n4102444800.0\n"), 0o600); err != nil {
			t.Fatal(err)
		}
	}
}

func TestExactPinnedHFSnapshotIsReusedOffline(t *testing.T) {
	const revision = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
	dir := t.TempDir()
	writeHFSnapshot(t, dir, revision)
	cfg := deploymentConfig("candle", dir)
	d := cfg.ModelDeployments["new"]
	d.Revision = revision
	cfg.ModelDeployments["new"] = d
	t.Setenv("PATH", t.TempDir())
	if err := EnsureModelsForConfig(cfg); err != nil {
		t.Fatalf("cached snapshot required network/CLI: %v", err)
	}
	if err := ValidateReloadArtifacts(cfg, cfg); err != nil {
		t.Fatal(err)
	}
}

func TestRetiredPinnedSnapshotCannotBeOverwritten(t *testing.T) {
	dir := t.TempDir()
	writeHFSnapshot(t, dir, "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
	before, err := os.ReadFile(filepath.Join(dir, "model.safetensors"))
	if err != nil {
		t.Fatal(err)
	}
	for _, revision := range []string{"", "main", "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"} {
		spec := ModelSpec{LocalPath: dir, Revision: revision, Strict: true}
		if validationErr := validateArtifactDownload(spec); validationErr == nil {
			t.Fatalf("retired snapshot accepted a write at revision %q", revision)
		}
	}
	after, err := os.ReadFile(filepath.Join(dir, "model.safetensors"))
	if err != nil {
		t.Fatal(err)
	}
	if string(before) != string(after) {
		t.Fatal("snapshot modified during validation")
	}
}

func TestLiveSnapshotCannotBeResyncedDuringReload(t *testing.T) {
	const revision = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
	dir := t.TempDir()
	writeHFSnapshot(t, dir, revision)
	cfg := deploymentConfig("candle", dir)
	d := cfg.ModelDeployments["new"]
	d.Revision = revision
	cfg.ModelDeployments["new"] = d
	if err := os.Remove(filepath.Join(dir, "tokenizer.json")); err != nil {
		t.Fatal(err)
	}
	if err := ValidateReloadArtifacts(cfg, cfg); err == nil {
		t.Fatal("live incomplete snapshot would be mutated")
	}
}

func TestReloadReusesUnversionedCompanionFromLiveSnapshot(t *testing.T) {
	const revision = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
	dir := t.TempDir()
	writeHFSnapshot(t, dir, revision, "labels.json")
	mapping := filepath.Join(dir, "labels.json")
	if err := os.WriteFile(mapping, []byte(`{"0":"billing"}`), 0o600); err != nil {
		t.Fatal(err)
	}
	current := deploymentConfig("candle", dir)
	current.ModelDeployments["new"] = config.ModelDeployment{Provider: "candle", Artifact: dir, Revision: revision}
	binding := current.ModelBindings["domain_classifier"]
	binding.MappingPath = mapping
	current.ModelBindings["domain_classifier"] = binding
	if err := ValidateReloadArtifacts(current, current); err != nil {
		t.Fatalf("current pinned snapshot is not complete: %v", err)
	}
	next := deploymentConfig("candle", dir)
	missingArtifact := filepath.Join(t.TempDir(), "unregistered-candidate")
	next.ModelDeployments["new"] = config.ModelDeployment{Provider: "candle", Artifact: missingArtifact, Revision: revision}
	next.ModelBindings["domain_classifier"] = binding

	// The existing mapping is still needed, but the candidate's revision does
	// not describe that separate snapshot. No download may touch the live path.
	specs, err := BuildModelSpecs(next)
	if err != nil {
		t.Fatal(err)
	}
	if len(specs) != 1 || specs[0].LocalPath != dir || !specs[0].FilesOnly || specs[0].Revision != "" {
		t.Errorf("expected an unversioned mapping companion, got %#v", specs)
	}
	if err := ValidateReloadArtifacts(current, next); err != nil {
		t.Errorf("complete live companion rejected before candidate preparation: %v", err)
	}
	t.Setenv("PATH", t.TempDir())
	if err := EnsureModelsForConfig(next); err != nil {
		t.Fatalf("read-only companion unexpectedly required the download CLI: %v", err)
	}
	if _, err := os.Stat(missingArtifact); !os.IsNotExist(err) {
		t.Fatalf("unregistered candidate was provisioned: %v", err)
	}
	if err := os.Remove(mapping); err != nil {
		t.Fatal(err)
	}
	if err := ValidateReloadArtifacts(current, next); err == nil {
		t.Fatal("missing live companion could be downloaded into the active snapshot")
	}
}

func TestReloadCompanionGraphRequiresExternalTensorFiles(t *testing.T) {
	const revision = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
	dir := t.TempDir()
	writeHFSnapshot(t, dir, revision, "model.onnx", "actual-weights.bin")
	entry := append(protoBytes(1, []byte("location")), protoBytes(2, []byte("actual-weights.bin"))...)
	graph := protoBytes(7, protoBytes(5, protoBytes(13, entry)))
	if err := os.WriteFile(filepath.Join(dir, "model.onnx"), graph, 0o600); err != nil {
		t.Fatal(err)
	}
	current := deploymentConfig("ort", dir)
	current.ModelDeployments["new"] = config.ModelDeployment{Provider: "ort", Artifact: dir, Revision: revision}
	binding := current.ModelBindings["domain_classifier"]
	binding.Head = filepath.Join(dir, "model.onnx")
	current.ModelBindings["domain_classifier"] = binding
	if err := ValidateReloadArtifacts(current, current); err != nil {
		t.Fatalf("current pinned graph is not complete: %v", err)
	}
	next := deploymentConfig("ort", dir)
	next.ModelDeployments["new"] = config.ModelDeployment{Provider: "ort", Artifact: filepath.Join(t.TempDir(), "candidate"), Revision: revision}
	next.ModelBindings["domain_classifier"] = binding
	if err := ValidateReloadArtifacts(current, next); err != nil {
		t.Errorf("complete external graph companion was rejected: %v", err)
	}
	if err := os.Remove(filepath.Join(dir, "actual-weights.bin")); err != nil {
		t.Fatal(err)
	}
	if err := ValidateReloadArtifacts(current, next); err == nil {
		t.Fatal("companion graph could download missing external tensors into the live snapshot")
	}
}

func TestReloadRevisionIntentPreservesLiveWriteProtection(t *testing.T) {
	const revision = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
	for _, test := range []struct {
		name     string
		revision string
		reject   bool
	}{
		{name: "same_pin", revision: revision},
		{name: "unspecified"},
		{name: "explicit_main", revision: "main", reject: true},
		{name: "different_pin", revision: "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb", reject: true},
	} {
		t.Run(test.name, func(t *testing.T) {
			dir := t.TempDir()
			writeHFSnapshot(t, dir, revision)
			current := deploymentConfig("candle", dir)
			current.ModelDeployments["new"] = config.ModelDeployment{Provider: "candle", Artifact: dir, Revision: revision}
			next := deploymentConfig("candle", dir)
			next.ModelDeployments["new"] = config.ModelDeployment{Provider: "candle", Artifact: dir, Revision: test.revision}
			specs, err := BuildModelSpecs(next)
			if err != nil {
				t.Fatal(err)
			}
			if len(specs) != 1 || specs[0].Revision != test.revision {
				t.Errorf("revision intent changed: %#v", specs)
			}
			if err := ValidateReloadArtifacts(current, next); (err != nil) != test.reject {
				t.Errorf("preflight error=%v, want rejection=%v", err, test.reject)
			}
			if !test.reject {
				t.Setenv("PATH", t.TempDir())
				if err := EnsureModelsForConfig(next); err != nil {
					t.Fatalf("complete snapshot required download: %v", err)
				}
			}
			if err := os.Remove(filepath.Join(dir, "tokenizer.json")); err != nil {
				t.Fatal(err)
			}
			if err := ValidateReloadArtifacts(current, next); err == nil {
				t.Fatal("revision intent allowed a live snapshot to be refreshed")
			}
		})
	}
}

func TestPinnedSnapshotRejectsStaleExtraProviderFiles(t *testing.T) {
	const revision = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
	dir := t.TempDir()
	writeHFSnapshot(t, dir, revision)
	if err := os.WriteFile(filepath.Join(dir, "model.onnx"), []byte("stale"), 0o600); err != nil {
		t.Fatal(err)
	}
	matched, err := cachedRevisionMatches(ModelSpec{LocalPath: dir, Revision: revision})
	if err != nil {
		t.Fatal(err)
	}
	if matched {
		t.Fatal("untracked old graph incorrectly matched pinned snapshot")
	}
}
