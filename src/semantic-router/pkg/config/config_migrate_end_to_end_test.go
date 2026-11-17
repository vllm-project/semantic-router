package config

import (
	"bytes"
	"errors"
	"fmt"
	"io/fs"
	"maps"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"gopkg.in/yaml.v3"
)

// migrateMaintainedDirs are the directories whose configurations the
// repository maintains: those the model resolution golden walks, the bench
// configurations and the other deployment assets.
var migrateMaintainedDirs = slices.Concat(maintainedConfigDirs, []string{
	"bench",
	filepath.Join("deploy", "kserve"),
	filepath.Join("deploy", "openshift"),
})

// migrateUnmaintainedDirs hold configurations that are no standalone
// configuration of this release, by why.
var migrateUnmaintainedDirs = map[string]string{
	"config/fragments":             "pieces of a configuration; the docs tests merge them where the pages show them",
	"config/recipes/built-in/v0.4": "the recipe bundle the 0.4 CLI installs, in config v0.3",
}

// The end-to-end migrate test: one migrate pass over every configuration the
// repository maintains (each recipe, e2e configuration and profile, and the
// deployment and bench assets) and over the configurations earlier releases
// shipped. A maintained configuration is canonical, so migrate leaves it as
// written, and a fragment that loads only where it is merged still writes no
// removed form; a release fixture fails to load with its hint and migrates to
// its golden, which loads and migrates to itself.
func TestMigrateEndToEnd(t *testing.T) {
	documents := maintainedConfigDocuments(t)
	// config/config.yaml is the reference configuration, router headers
	// included.
	if reference, ok := documents["config/config.yaml"]; !ok || !bytes.Contains(reference, []byte("session_id_headers:")) {
		t.Fatal("config/config.yaml, with global.router.headers.session_id_headers, is not among the maintained configurations")
	}
	for _, key := range slices.Sorted(maps.Keys(documents)) {
		data := documents[key]
		t.Run(key, func(t *testing.T) {
			if _, err := ParseYAMLBytesWithoutEnvExpansion(data); err != nil {
				if strings.Contains(err.Error(), "run `vllm-sr config migrate`") {
					t.Fatalf("a maintained configuration is written in a removed form: %v", err)
				}
				return
			}
			migrated, notes, err := MigrateConfig(data)
			if err != nil || len(notes) > 0 || !bytes.Equal(migrated, data) {
				t.Fatalf("migrate rewrote a maintained configuration: %v %+v\n%s", err, notes, migrated)
			}
		})
	}

	releases, err := filepath.Glob(filepath.Join("testdata", "migrate_releases", "*", "before.yaml"))
	if err != nil || len(releases) == 0 {
		t.Fatalf("no release fixtures: %v", err)
	}
	for _, beforePath := range releases {
		dir := filepath.Dir(beforePath)
		t.Run("release/"+filepath.Base(dir), func(t *testing.T) {
			migrateReleaseFixture(t, dir, readGolden(t, beforePath))
		})
	}
}

// migrateReleaseFixture checks one configuration an earlier release shipped.
// One written in a form with no mechanical rewrite has refused.txt instead of
// after.yaml: the line migrate refuses it with.
func migrateReleaseFixture(t *testing.T, dir string, before []byte) {
	t.Helper()
	hintPath, afterPath := filepath.Join(dir, "hint.txt"), filepath.Join(dir, "after.yaml")
	original, loadErr := ParseYAMLBytesWithoutEnvExpansion(before)
	lines := formLines(original, loadErr)
	if *updateMigrateGoldens {
		if len(lines) == 0 {
			t.Fatal("the release fixture loads; it is no legacy configuration")
		}
		if err := os.WriteFile(hintPath, []byte(lines[0]+"\n"), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	if hint := strings.TrimSpace(string(readGolden(t, hintPath))); !containsLine(lines, hint) {
		t.Fatalf("loading before.yaml reports\n%s\nwant the line\n%s", strings.Join(lines, "\n"), hint)
	}
	migrated, _, err := MigrateConfig(before)
	if refusal, readErr := os.ReadFile(filepath.Join(dir, "refused.txt")); readErr == nil {
		if err == nil || !containsLine(strings.Split(err.Error(), "\n"), strings.TrimSpace(string(refusal))) {
			t.Fatalf("migrate refused with\n%v\nwant the line\n%s", err, refusal)
		}
		return
	}
	if err != nil {
		t.Fatalf("migrate: %v", err)
	}
	if *updateMigrateGoldens {
		if err = os.WriteFile(afterPath, migrated, 0o600); err != nil {
			t.Fatal(err)
		}
	}
	if want := readGolden(t, afterPath); !bytes.Equal(migrated, want) {
		t.Fatalf("migrate wrote\n%s\nwant (after.yaml; -update-migrate rewrites it)\n%s", migrated, want)
	}
	again, notes, err := MigrateConfig(migrated)
	if err != nil || len(notes) > 0 || !bytes.Equal(again, migrated) {
		t.Fatalf("migrating after.yaml again rewrote it: %v %+v", err, notes)
	}
}

// maintainedConfigDocuments returns every Router configuration the
// maintained directories hold, by path (and #index for a later YAML
// document of a file): whole files, the config a Helm or e2e values file
// embeds, and the config.yaml a ConfigMap carries.
func maintainedConfigDocuments(t *testing.T) map[string][]byte {
	t.Helper()
	documents := map[string][]byte{}
	repo := os.DirFS(maintainedRepoPrefix)
	for _, dir := range migrateMaintainedDirs {
		err := fs.WalkDir(repo, dir, func(path string, entry fs.DirEntry, err error) error {
			if err == nil && path != dir && strings.HasPrefix(entry.Name(), ".") {
				if entry.IsDir() {
					return fs.SkipDir
				}
				return nil
			}
			if _, unmaintained := migrateUnmaintainedDirs[filepath.ToSlash(path)]; unmaintained && entry.IsDir() {
				return fs.SkipDir
			}
			if err != nil || entry.IsDir() || (!strings.HasSuffix(path, ".yaml") && !strings.HasSuffix(path, ".yml")) {
				return err
			}
			data, err := fs.ReadFile(repo, path)
			if err != nil {
				return err
			}
			for index, document := range strings.Split(string(data), "\n---\n") {
				if modelCatalogDocument([]byte(document)) {
					continue
				}
				config, ok := routerConfigDocument([]byte(document))
				if !ok {
					config, ok = configMapRouterConfig([]byte(document))
				}
				if !ok {
					continue
				}
				key := filepath.ToSlash(path)
				if index > 0 {
					key = fmt.Sprintf("%s#%d", key, index)
				}
				documents[key] = config
			}
			return nil
		})
		if err != nil && !errors.Is(err, fs.ErrNotExist) {
			t.Fatal(err)
		}
	}
	return documents
}

// modelCatalogDocument reports whether a document is a recipe or model
// catalog (schema_version vllm-sr/model-catalog/...), not a configuration.
func modelCatalogDocument(data []byte) bool {
	var probe struct {
		SchemaVersion string `yaml:"schema_version"`
	}
	return yaml.Unmarshal(data, &probe) == nil && strings.HasPrefix(probe.SchemaVersion, "vllm-sr/model-catalog/")
}

// configMapRouterConfig returns the config.yaml a ConfigMap carries.
func configMapRouterConfig(data []byte) ([]byte, bool) {
	var probe struct {
		Kind string            `yaml:"kind"`
		Data map[string]string `yaml:"data"`
	}
	if yaml.Unmarshal(data, &probe) != nil || probe.Kind != "ConfigMap" || probe.Data["config.yaml"] == "" {
		return nil, false
	}
	config, ok := routerConfigDocument([]byte(probe.Data["config.yaml"]))
	return config, ok
}
