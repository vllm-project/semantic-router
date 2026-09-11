package main

import (
	"os"
	"path/filepath"
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
