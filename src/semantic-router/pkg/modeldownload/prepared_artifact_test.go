package modeldownload

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// This synthetic inventory tests offline provisioning only. Typed native tests
// independently execute tiny ONNX graphs and reject incorrect tensor contracts.
func preparedFixture(t *testing.T, root string, spec ModelSpec) {
	t.Helper()
	manifest := preparedManifest{FormatVersion: 1, Adapter: "vela_omni", Variant: "nano", Tokenizer: "text/tokenizer.json"}
	manifest.Source.RepoID, manifest.Source.Revision = spec.RepoID, spec.Revision
	manifest.Processors.Audio.File = "processors/audio.json"
	manifest.ReferenceParity.File, manifest.ReferenceParity.Passed = "reference_parity.json", true
	manifest.Graphs = map[string]struct {
		File string `json:"file"`
	}{}
	files := map[string][]byte{"text/tokenizer.json": []byte("tokenizer fixture"), "processors/audio.json": []byte("processor fixture")}
	for _, name := range []string{"text", "image", "clap", "audio"} {
		file := "onnx/" + name + ".onnx"
		manifest.Graphs[name] = struct {
			File string `json:"file"`
		}{file}
		files[file] = []byte("tensor fixture " + name)
	}
	report, err := json.Marshal(map[string]any{"fixture_only": true, "passed": true, "variant": "nano", "source": manifest.Source, "tests": []any{map[string]any{"passed": true}}})
	if err != nil {
		t.Fatal(err)
	}
	files[manifest.ReferenceParity.File] = report
	manifest.Files = map[string]string{}
	for file, data := range files {
		path := filepath.Join(root, file)
		if mkdirErr := os.MkdirAll(filepath.Dir(path), 0o755); mkdirErr != nil {
			t.Fatal(mkdirErr)
		}
		if writeErr := os.WriteFile(path, data, 0o600); writeErr != nil {
			t.Fatal(writeErr)
		}
		digest := sha256.Sum256(data)
		manifest.Files[file] = hex.EncodeToString(digest[:])
	}
	data, err := json.Marshal(manifest)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, omniManifestName), data, 0o600); err != nil {
		t.Fatal(err)
	}
}

func fixtureOmniSpec(path string) ModelSpec {
	catalog := config.GetModelByPath("omni-nano")
	return ModelSpec{LocalPath: path, RepoID: catalog.RepoID, Revision: catalog.Revision, PreparedArtifact: catalog.PreparedArtifact, ArtifactBundle: catalog.ArtifactBundle, Strict: true}
}

func TestPreparedArtifactFreshCacheDoesNotNeedHuggingFaceCLI(t *testing.T) {
	root := t.TempDir()
	t.Chdir(root)
	t.Setenv("PATH", "")
	t.Setenv("ROUTER_MODEL_ARTIFACTS", filepath.Join(root, "bundles"))
	spec := fixtureOmniSpec("models/vela-1.0-omni-nano")
	preparedFixture(t, filepath.Join(root, "bundles", spec.ArtifactBundle), spec)
	cfg := &config.RouterConfig{MoMRegistry: config.ToLegacyRegistry()}
	cfg.MultiModalModelPath = spec.LocalPath
	cfg.EmbeddingConfig.ModelType = "multimodal"
	cfg.API.Embeddings.Enabled = true
	specs, err := BuildModelSpecs(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if len(specs) != 1 || specs[0].PreparedArtifact != "vela_omni" || specs[0].Revision != spec.Revision || len(specs[0].RequiredFiles) != 0 || len(specs[0].RequiredFileGroups) != 0 {
		t.Fatalf("native weights leaked into prepared inventory: %+v", specs)
	}
	if err := EnsureModelsForConfig(cfg); err != nil {
		t.Fatal(err)
	}
	if missing, err := GetMissingModels(specs); err != nil || len(missing) != 0 {
		t.Fatalf("prepared cache requires HF revision metadata: %+v %v", missing, err)
	}
	// A prepared cache remains usable without its image source bundle.
	t.Setenv("ROUTER_MODEL_ARTIFACTS", filepath.Join(root, "absent"))
	if err := EnsureModelsForConfig(cfg); err != nil {
		t.Fatal(err)
	}
}

func TestPreparedArtifactRejectsCorruptionSourceMismatchAndEscapes(t *testing.T) {
	for _, kind := range []string{"digest", "revision", "escape", "symlink", "receipt"} {
		t.Run(kind, func(t *testing.T) {
			spec := fixtureOmniSpec(t.TempDir())
			preparedFixture(t, spec.LocalPath, spec)
			path := filepath.Join(spec.LocalPath, omniManifestName)
			data, _ := os.ReadFile(path)
			var manifest preparedManifest
			if err := json.Unmarshal(data, &manifest); err != nil {
				t.Fatal(err)
			}
			switch kind {
			case "digest":
				if err := os.WriteFile(filepath.Join(spec.LocalPath, "onnx/text.onnx"), []byte("corrupted"), 0o600); err != nil {
					t.Fatal(err)
				}
			case "revision":
				manifest.Source.Revision = strings.Repeat("f", 40)
			case "escape":
				manifest.Files["../outside"] = strings.Repeat("0", 64)
			case "symlink":
				file := filepath.Join(spec.LocalPath, "onnx/text.onnx")
				if err := os.Remove(file); err != nil {
					t.Fatal(err)
				}
				if err := os.Symlink(t.TempDir(), file); err != nil {
					t.Fatal(err)
				}
			case "receipt":
				manifest.ReferenceParity.Passed = false
			}
			data, _ = json.Marshal(manifest)
			if err := os.WriteFile(path, data, 0o600); err != nil {
				t.Fatal(err)
			}
			if _, err := GetMissingModels([]ModelSpec{spec}); err == nil {
				t.Fatal("invalid prepared artifact accepted")
			}
		})
	}
}

func TestPreparedArtifactMissingBundleFailsWithoutNativeFallback(t *testing.T) {
	t.Setenv("ROUTER_MODEL_ARTIFACTS", t.TempDir())
	spec := fixtureOmniSpec(filepath.Join(t.TempDir(), "cache"))
	err := DownloadModelWithProgressContext(context.Background(), spec, DownloadConfig{})
	if err == nil || !strings.Contains(err.Error(), "VELA_OMNI_VARIANTS") {
		t.Fatalf("missing bundle must explain preparation: %v", err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if err := provisionPreparedArtifact(ctx, spec); !errors.Is(err, context.Canceled) {
		t.Fatalf("cancel = %v", err)
	}
	if _, err := os.Stat(spec.LocalPath); !os.IsNotExist(err) {
		t.Fatalf("failed preparation published partial cache: %v", err)
	}
}

func TestPublishedOmniPreparedInventory(t *testing.T) {
	root := os.Getenv("VELA_OMNI_ARTIFACT")
	if root == "" {
		t.Skip("set VELA_OMNI_ARTIFACT to a prepared pinned release")
	}
	data, err := os.ReadFile(filepath.Join(root, omniManifestName))
	if err != nil {
		t.Fatal(err)
	}
	var manifest preparedManifest
	if err := json.Unmarshal(data, &manifest); err != nil {
		t.Fatal(err)
	}
	catalog := config.GetModelByPath("omni-" + manifest.Variant)
	if catalog == nil {
		t.Fatal("unknown published variant")
	}
	spec := ModelSpec{LocalPath: root, RepoID: catalog.RepoID, Revision: catalog.Revision, PreparedArtifact: catalog.PreparedArtifact, ArtifactBundle: catalog.ArtifactBundle, Strict: true}
	if missing, err := GetMissingModels([]ModelSpec{spec}); err != nil || len(missing) > 0 {
		t.Fatalf("prepared release failed provisioning validation: %v %v", missing, err)
	}
}
