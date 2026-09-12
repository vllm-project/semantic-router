package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestGateRequiresSelectedMatrix(t *testing.T) {
	for _, tc := range []struct {
		name                        string
		positive, negative, shipped float64
		want                        int
	}{
		{"crossed boundary", 0.50004, 0.5, 0.5, 2},
		{"float32 rounding", 0.7, 0.52, 0.61, 0},
	} {
		t.Run(tc.name, func(t *testing.T) {
			fixtures := []fixtureReport{
				{Path: "p", PositiveFor: []string{testRule}, Scores: map[string]float64{testRule: tc.positive}},
				{Path: "n", Scores: map[string]float64{testRule: tc.negative}},
			}
			rule := calibrateRule(config.EmbeddingRule{Name: testRule, SimilarityThreshold: float32(tc.shipped)}, fixtures)
			report := calibrationReport{Fixtures: fixtures, Rules: []ruleReport{rule}}
			if got := checkShippedThresholds(report, "pin", "pin", true); got != tc.want {
				t.Fatalf("gate=%d, want %d; shipped=%+v selected=%+v", got, tc.want, rule.Shipped, rule.Selected)
			}
		})
	}
}

// A dirty worktree is a warning locally and a failure under -require-clean,
// independent of the thresholds matching.
func TestGateDirtyWorktree(t *testing.T) {
	fixtures := []fixtureReport{
		{Path: "p", PositiveFor: []string{testRule}, Scores: map[string]float64{testRule: 0.7}},
		{Path: "n", Scores: map[string]float64{testRule: 0.52}},
	}
	rule := calibrateRule(config.EmbeddingRule{Name: testRule, SimilarityThreshold: 0.61}, fixtures)
	report := calibrationReport{Fixtures: fixtures, Rules: []ruleReport{rule}}
	report.Source.Dirty = true
	if got := checkShippedThresholds(report, "pin", "pin", false); got != 0 {
		t.Fatalf("dirty worktree without -require-clean returned %d, want 0 (warning)", got)
	}
	if got := checkShippedThresholds(report, "pin", "pin", true); got != 2 {
		t.Fatalf("dirty worktree with -require-clean returned %d, want 2", got)
	}
}

// The scoring provenance the report carries must reach both outputs.
func TestReportCarriesScoringProvenance(t *testing.T) {
	hnsw := classifierConfig()
	report := calibrationReport{}
	report.Model.TargetDimension = hnsw.TargetDimension
	report.Model.TargetLayer = hnsw.TargetLayer
	report.Model.Scoring = hnsw.PrototypeScoring
	report.Model.ArtifactFiles = map[string]string{"model.safetensors": "sha256:abc"}
	data, err := json.Marshal(report)
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{`"target_layer":0`, `"target_dimension":384`, `"artifact_files":{"model.safetensors":"sha256:abc"}`} {
		if !strings.Contains(string(data), want) {
			t.Errorf("JSON report lacks %s", want)
		}
	}
	markdown := renderMarkdown(report)
	for _, want := range []string{"Target layer (candidate text embeddings): `0 (final layer)`", "Model file `model.safetensors`: `sha256:abc`"} {
		if !strings.Contains(markdown, want) {
			t.Errorf("Markdown report lacks %q", want)
		}
	}
}

func TestRoundTwoPlacesSigned(t *testing.T) {
	for input, want := range map[float64]float64{-0.516: -0.52, 0.516: 0.52, -0.514: -0.51, 0.514: 0.51} {
		if got := roundTwoPlaces(input); got != want {
			t.Errorf("round(%v)=%v, want %v", input, got, want)
		}
	}
}

func TestCanonicalRepoRootResolvesAliasBeforeParent(t *testing.T) {
	a, b := gitRepo(t), gitRepo(t)
	if err := os.Mkdir(filepath.Join(b, "sub"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(filepath.Join(b, "sub"), filepath.Join(a, "alias")); err != nil {
		t.Fatal(err)
	}
	want, err := filepath.EvalSymlinks(b)
	if err != nil {
		t.Fatal(err)
	}
	got, err := canonicalRepoRoot(a + "/alias/..")
	if err != nil || got != want {
		t.Fatalf("root=%q err=%v, want %q", got, err, want)
	}
	if err := bindToCommit(got, []string{"img/tracked.png"}); err != nil {
		t.Fatal(err)
	}
	if _, err := canonicalRepoRoot(filepath.Join(b, "sub")); err == nil {
		t.Fatal("accepted a subtree as the repository root")
	}
}

func TestValidateOutputsProtectsInputsAndTrackedFiles(t *testing.T) {
	root, err := canonicalRepoRoot(gitRepo(t))
	if err != nil {
		t.Fatal(err)
	}
	input := filepath.Join(root, "img", "tracked.png")
	if err := os.WriteFile(input, []byte("modified"), 0o644); err != nil {
		t.Fatal(err)
	}
	generated := filepath.Join(root, "report.json")
	for name, outputs := range map[string][]string{
		"input overlap":      {input},
		"other tracked file": {filepath.Join(root, ".gitignore")},
		"duplicate":          {generated, generated},
		"symlink":            {filepath.Join(root, "img", "link.png")},
		"directory":          {filepath.Join(root, "img")},
	} {
		t.Run(name, func(t *testing.T) {
			if _, err := validateOutputs(root, []string{"img/tracked.png"}, outputs); err == nil {
				t.Fatal("accepted unsafe report destination")
			}
		})
	}
	if _, err := validateOutputs(root, []string{"img/tracked.png"}, []string{generated, filepath.Join(t.TempDir(), "report.md")}); err != nil {
		t.Fatalf("generated outputs rejected: %v", err)
	}
}

func TestRepoStateExcludesOnlyLiteralOutput(t *testing.T) {
	root := gitRepo(t)
	for _, name := range []string{"img/untracked.png", "img/link.png"} {
		if err := os.Remove(filepath.Join(root, name)); err != nil {
			t.Fatal(err)
		}
	}
	output := filepath.Join(root, "img", "*.png")
	if err := os.WriteFile(output, []byte("report"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, dirty := repoState(root, output); dirty {
		t.Fatal("generated report should be excluded")
	}
	if err := os.WriteFile(filepath.Join(root, "img", "tracked.png"), []byte("changed"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, dirty := repoState(root, output); !dirty {
		t.Fatal("output wildcard hid a modified tracked input")
	}
}
