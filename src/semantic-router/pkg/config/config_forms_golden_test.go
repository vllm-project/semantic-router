package config

import (
	"flag"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"gopkg.in/yaml.v3"
)

var updateMigrateGoldens = flag.Bool("update-migrate", false, "rewrite testdata/migrate/*/after.yaml from the migration")

// Every removed and renamed form has a golden case under testdata/migrate:
// before.yaml loaded directly fails (removed) or warns (renamed) with the
// line in hint.txt; `vllm-sr config migrate` writes after.yaml, which loads
// with no form warning, or refuses with the line in refused.txt; and a
// renamed form's effective configuration does not change.
func TestMigrateGoldenCases(t *testing.T) {
	cases, err := filepath.Glob(filepath.Join("testdata", "migrate", "*", "before.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	covered := make([]bool, len(configForms))
	for _, beforePath := range cases {
		dir := filepath.Dir(beforePath)
		t.Run(filepath.Base(dir), func(t *testing.T) {
			before := readGolden(t, beforePath)
			markCoveredForms(t, before, covered)
			hint := strings.TrimSpace(string(readGolden(t, filepath.Join(dir, "hint.txt"))))
			original, loadErr := ParseYAMLBytesWithoutEnvExpansion(before)
			if lines := formLines(original, loadErr); !containsLine(lines, hint) {
				t.Fatalf("loading before.yaml reports\n%s\nwant the line\n%s", strings.Join(lines, "\n"), hint)
			}
			// A form with no mechanical rewrite has refused.txt instead of
			// after.yaml: the line migrate refuses with.
			if refusal, err := os.ReadFile(filepath.Join(dir, "refused.txt")); err == nil {
				_, _, migrateErr := MigrateConfig(before)
				if migrateErr == nil || !containsLine(strings.Split(migrateErr.Error(), "\n"), strings.TrimSpace(string(refusal))) {
					t.Fatalf("migrate refused with\n%v\nwant the line\n%s", migrateErr, refusal)
				}
				return
			}
			migrated, _, err := MigrateConfig(before)
			if err != nil {
				t.Fatalf("migrate: %v", err)
			}
			afterPath := filepath.Join(dir, "after.yaml")
			if *updateMigrateGoldens {
				if err = os.WriteFile(afterPath, migrated, 0o600); err != nil {
					t.Fatal(err)
				}
			}
			if want := readGolden(t, afterPath); string(migrated) != string(want) {
				t.Fatalf("migrate wrote\n%s\nwant (after.yaml; -update-migrate rewrites it)\n%s", migrated, want)
			}
			rewritten, err := ParseYAMLBytesWithoutEnvExpansion(migrated)
			if err != nil {
				t.Fatalf("after.yaml does not load: %v", err)
			}
			if len(rewritten.FormWarnings) > 0 {
				t.Fatalf("after.yaml is still written in a renamed form: %+v", rewritten.FormWarnings)
			}
			if loadErr == nil && !reflect.DeepEqual(effectiveConfig(t, original), effectiveConfig(t, rewritten)) {
				t.Fatalf("the renamed form and its rewrite configure different things:\n%s\n---\n%s", effectiveConfig(t, original), effectiveConfig(t, rewritten))
			}
		})
	}
	for i, form := range configForms {
		if !covered[i] {
			t.Errorf("form %s at %v has no golden case under testdata/migrate", form.Code, form.Paths)
		}
	}
}

func readGolden(t *testing.T, path string) []byte {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	return data
}

func markCoveredForms(t *testing.T, data []byte, covered []bool) {
	t.Helper()
	var document yaml.Node
	if err := yaml.Unmarshal(data, &document); err != nil {
		t.Fatal(err)
	}
	if root := documentRoot(&document); root != nil {
		for i, form := range configForms {
			covered[i] = covered[i] || len(form.matches(root)) > 0
		}
	}
}

// formLines are the one-line errors of a load, or the warnings of the
// renamed forms it read.
func formLines(cfg *RouterConfig, err error) []string {
	if err != nil {
		return strings.Split(err.Error(), "\n")
	}
	var lines []string
	for _, warning := range cfg.FormWarnings {
		lines = append(lines, warning.Field+": "+warning.Code+": "+warning.Message)
	}
	return lines
}

func containsLine(lines []string, line string) bool {
	for _, candidate := range lines {
		if strings.TrimSpace(candidate) == line {
			return true
		}
	}
	return false
}

func effectiveConfig(t *testing.T, cfg *RouterConfig) string {
	t.Helper()
	data, err := yaml.Marshal(CanonicalConfigFromRouterConfig(cfg))
	if err != nil {
		t.Fatal(err)
	}
	return string(data)
}
