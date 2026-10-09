package modeldownload

import (
	"os"
	"path/filepath"
	"slices"
	"testing"

	"google.golang.org/protobuf/encoding/protowire"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

const deploymentRevision = "abcabcabcabcabcabcabcabcabcabcabcabcabca"

// deploymentConfig binds the domain classifier to a model_runtime deployment
// of artifact, pinned when artifact is a repository.
func deploymentConfig(artifact string) *config.RouterConfig {
	cfg := &config.RouterConfig{MoMRegistry: map[string]string{"models/old": "test/old", artifact: "test/new"}}
	deployment := config.ModelDeployment{Provider: config.ModelRuntimeProvider, Artifact: artifact}
	if !filepath.IsAbs(artifact) {
		deployment.Revision = deploymentRevision
	}
	cfg.ModelDeployments = map[string]config.ModelDeployment{"new": deployment}
	cfg.CategoryModel.ModelID = "models/old"
	cfg.CategoryMappingPath = "labels.json"
	cfg.ModelBindings = map[string]config.ModelBinding{"domain_classifier": {Deployment: "new", Contract: "label_distribution.v1", Adapter: "mmbert32k"}}
	cfg.Decisions = []config.Decision{{Name: "route", Rules: config.RuleNode{Type: config.SignalTypeDomain, Name: "billing"}}}
	return cfg
}

// The runtime downloads a bound deployment's artifact; the router fetches only
// the binding's label map, at the release the runtime serves.
func TestBoundDeploymentProvisionsOnlyItsMappingAtItsRelease(t *testing.T) {
	cfg := deploymentConfig("models/new")
	binding := cfg.ModelBindings["domain_classifier"]
	binding.MappingPath = "models/new/labels.json"
	cfg.ModelBindings["domain_classifier"] = binding
	specs, err := BuildModelSpecs(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if len(specs) != 1 || specs[0].LocalPath != "models/new" || specs[0].Revision != deploymentRevision || !specs[0].FilesOnly ||
		!slices.Equal(specs[0].RequiredFiles, []string{"labels.json"}) {
		t.Fatalf("specs=%#v", specs)
	}
	if cfg.CategoryModel.ModelID != "models/old" {
		t.Fatal("source configuration mutated")
	}
}

func TestUnregisteredLocalArtifactsDoNotRequireRegistry(t *testing.T) {
	cfg := deploymentConfig(filepath.Join(t.TempDir(), "custom-model"))
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

func TestBoundMappingInASeparateSnapshotIsFilesOnly(t *testing.T) {
	cfg := deploymentConfig("models/backbone")
	cfg.MoMRegistry["models/mappings"] = "test/maps"
	binding := cfg.ModelBindings["domain_classifier"]
	binding.MappingPath = "models/mappings/domain.json"
	cfg.ModelBindings["domain_classifier"] = binding
	specs, err := BuildModelSpecs(cfg)
	if err != nil {
		t.Fatal(err)
	}
	mapping, ok := findSpecByPath(specs, "models/mappings")
	if len(specs) != 1 || !ok || !mapping.FilesOnly || !slices.Contains(mapping.RequiredFiles, "domain.json") || mapping.Revision != "" {
		t.Fatalf("specs=%#v", specs)
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
	cfg := deploymentConfig("models/default")
	cfg.ModelDeployments["private"] = config.ModelDeployment{Provider: config.ModelRuntimeProvider, Artifact: "models/private"}
	cfg.ModelDeployments["dormant"] = config.ModelDeployment{Provider: config.ModelRuntimeProvider, Artifact: "models/dormant"}
	cfg.MoMRegistry["models/private"] = "test/private"
	cfg.MoMRegistry["models/dormant"] = "test/dormant"
	profile := func(deployment string) config.RoutingProfile {
		artifact := cfg.ModelDeployments[deployment].Artifact
		return config.RoutingProfile{ModelBindings: map[string]config.ModelBinding{"domain_classifier": {Deployment: deployment, Contract: "label_distribution.v1", Adapter: "mmbert32k", MappingPath: artifact + "/labels.json"}}, Decisions: cfg.Decisions}
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
	spec := ModelSpec{LocalPath: dir, Revision: revision}
	for _, name := range append([]string{"config.json", "tokenizer.json", "model.safetensors"}, extraFiles...) {
		writeHFRevisionArtifact(t, spec, name, "fixture", true)
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

func TestGlobalServiceDownloadsIgnoreRecipeOverrides(t *testing.T) {
	for _, explicit := range []bool{false, true} {
		t.Run(map[bool]string{false: "module_defaults", true: "global_bindings"}[explicit], func(t *testing.T) {
			cfg := &config.RouterConfig{MoMRegistry: map[string]string{
				"models/global-embedding": "test/global-embedding",
				"models/recipe-embedding": "test/recipe-embedding",
			}}
			cfg.MmBertModelPath = "models/global-embedding"
			cfg.EmbeddingConfig.ModelType = "mmbert"
			cfg.EmbeddingModels.UseCPU = true
			cfg.Tools.Enabled = true
			cfg.SemanticCache.Enabled = true
			cfg.SemanticCache.EmbeddingModel = "mmbert"
			cfg.ModelDeployments = map[string]config.ModelDeployment{
				"global-embedding": {Provider: config.ModelRuntimeProvider, Device: "cpu", Artifact: "models/global-embedding"},
				"recipe-embedding": {Provider: config.ModelRuntimeProvider, Device: "rocm:7", Artifact: "models/recipe-embedding"},
			}
			cfg.ModelBindings = map[string]config.ModelBinding{
				"embedding": {Deployment: "recipe-embedding", Adapter: "mmbert", Contract: "embedding.v1"},
			}
			if explicit {
				cfg.GlobalModelBindings = map[string]config.ModelBinding{
					"embedding": {Deployment: "global-embedding", Adapter: "mmbert", Contract: "embedding.v1"},
				}
			}
			specs, err := BuildModelSpecs(cfg)
			if err != nil {
				t.Fatal(err)
			}
			got := map[string]bool{}
			for _, spec := range specs {
				got[spec.LocalPath] = true
			}
			// The global binding's runtime downloads its own model.
			want := map[string]bool{"models/global-embedding": !explicit}
			if got["models/recipe-embedding"] || got["models/global-embedding"] != want["models/global-embedding"] || len(got) > 1 {
				t.Fatalf("service download used recipe source: %+v", specs)
			}
			cfg.Tools.Enabled = false
			cfg.Decisions = []config.Decision{{Name: "no-cache-consumer"}}
			specs, err = BuildModelSpecs(cfg)
			if err != nil {
				t.Fatal(err)
			}
			if len(specs) != 0 {
				t.Fatalf("idle global models provisioned: %+v", specs)
			}
		})
	}
}
