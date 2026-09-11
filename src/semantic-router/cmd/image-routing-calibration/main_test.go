package main

import (
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

const testRule = "rule"

// scoredFixtures builds one fixture per score; the positive set is by path.
func scoredFixtures(scores map[string]float64, positives ...string) ([]fixtureReport, map[string]bool) {
	fixtures := make([]fixtureReport, 0, len(scores))
	for path, score := range scores {
		fixtures = append(fixtures, fixtureReport{Path: path, Scores: map[string]float64{testRule: score}})
	}
	positive := map[string]bool{}
	for _, path := range positives {
		positive[path] = true
	}
	return fixtures, positive
}

func calibrate(t *testing.T, fixtures []fixtureReport, positive map[string]bool) ruleReport {
	t.Helper()
	for i := range fixtures {
		if positive[fixtures[i].Path] {
			fixtures[i].PositiveFor = []string{testRule}
		}
	}
	return calibrateRule(config.EmbeddingRule{Name: testRule, SimilarityThreshold: 0.5}, fixtures)
}

// Two max-F1 bands with equal F1 but different widths: (0.30, 0.50] is 0.20
// wide, (0.85, 0.90] is 0.05 wide. The global margin would prefer the narrow
// band (it sits closer to max(negative)); the documented rule prefers the
// wider one, so the midpoint 0.40 must win.
func TestSelectThreshold_TieBreaksTowardWiderBand(t *testing.T) {
	fixtures, positive := scoredFixtures(map[string]float64{
		"neg-a": 0.10, "neg-b": 0.30, "pos-a": 0.50, "neg-c": 0.70, "neg-d": 0.85, "pos-b": 0.90,
	}, "pos-a", "pos-b")
	report := calibrate(t, fixtures, positive)

	if report.Selected.Separable {
		t.Fatalf("fixture set is interleaved; expected non-separable, got %+v", report.Selected)
	}
	if got, want := report.Selected.Threshold, 0.40; math.Abs(got-want) > 1e-9 {
		t.Fatalf("selected threshold = %.4f, want %.4f (midpoint of the wider max-F1 band)", got, want)
	}
	if got, want := report.Selected.F1, 2.0/3.0; math.Abs(got-want) > 1e-9 {
		t.Fatalf("selected F1 = %.4f, want %.4f", got, want)
	}
}

// Separable sets take the midpoint of the gap, rounded to two decimals when
// the rounded value still separates.
func TestSelectThreshold_SeparableMidpointRounds(t *testing.T) {
	fixtures, positive := scoredFixtures(map[string]float64{
		"neg-a": 0.20, "neg-b": 0.41, "pos-a": 0.63, "pos-b": 0.80,
	}, "pos-a", "pos-b")
	report := calibrate(t, fixtures, positive)

	if !report.Selected.Separable {
		t.Fatalf("expected separable selection, got %+v", report.Selected)
	}
	if got, want := report.Selected.Threshold, 0.52; math.Abs(got-want) > 1e-9 {
		t.Fatalf("selected threshold = %.4f, want %.4f (rounded gap midpoint)", got, want)
	}
}

// The sweep must not depend on the currently shipped threshold.
func TestCalibrateRule_SweepIgnoresShippedThreshold(t *testing.T) {
	fixtures, positive := scoredFixtures(map[string]float64{
		"neg-a": 0.10, "neg-b": 0.30, "pos-a": 0.50, "neg-c": 0.70, "neg-d": 0.85, "pos-b": 0.90,
	}, "pos-a", "pos-b")
	for i := range fixtures {
		if positive[fixtures[i].Path] {
			fixtures[i].PositiveFor = []string{testRule}
		}
	}
	a := calibrateRule(config.EmbeddingRule{Name: testRule, SimilarityThreshold: 0.44}, fixtures)
	b := calibrateRule(config.EmbeddingRule{Name: testRule, SimilarityThreshold: 0.88}, fixtures)
	if a.Selected.Threshold != b.Selected.Threshold || len(a.Sweep) != len(b.Sweep) {
		t.Fatalf("selection changed with the shipped value: %.4f (sweep %d) vs %.4f (sweep %d)",
			a.Selected.Threshold, len(a.Sweep), b.Selected.Threshold, len(b.Sweep))
	}
}

// Two bands with different confusion matrices but the same mathematical F1
// (1/3 each). Through precision and recall the floats differ in the last bit,
// so a float comparison never reaches the width tie-break and picks the
// narrow (0, 0.1] band; the rational comparison must pick (0.15, 0.7].
func TestSelectThreshold_TieUsesExactF1(t *testing.T) {
	fixtures, positive := scoredFixtures(map[string]float64{
		"p1": 0.1, "p2": 0.7,
		"n1": 0.11, "n2": 0.12, "n3": 0.13, "n4": 0.14, "n5": 0.15, "n6": 0.8, "n7": 0.85, "n8": 0.9,
	}, "p1", "p2")
	report := calibrate(t, fixtures, positive)

	// The wider band is (0.15, 0.7]; its midpoint 0.425 rounds to 0.42, which
	// yields the same matrix and is therefore what the selector reports.
	if got := report.Selected.Threshold; got <= 0.15 || got > 0.7 {
		t.Fatalf("selected threshold = %.4f, want a value inside the wider equal-F1 band (0.15, 0.7]", got)
	}
	if report.Selected.TP != 1 || report.Selected.FP != 3 || report.Selected.FN != 1 {
		t.Fatalf("selected matrix = TP %d FP %d FN %d, want 1/3/1", report.Selected.TP, report.Selected.FP, report.Selected.FN)
	}
}

// A positive scoring exactly at the floor is only admitted by the boundary
// threshold itself; the band midpoints all miss it.
func TestSelectThreshold_ZeroScoredPositiveUsesBoundary(t *testing.T) {
	fixtures, positive := scoredFixtures(map[string]float64{"p1": 0, "n1": 0.5}, "p1")
	report := calibrate(t, fixtures, positive)

	if report.Selected.Threshold != 0 || report.Selected.TP != 1 || report.Selected.FP != 1 {
		t.Fatalf("selected = %+v, want threshold 0 with TP 1 FP 1", report.Selected)
	}
}

// One distinct score means a one-value sweep; the reported matrix must be the
// evaluation of that threshold, not an empty result.
func TestSelectThreshold_SingletonSweepIsEvaluated(t *testing.T) {
	fixtures, positive := scoredFixtures(map[string]float64{"p1": 0, "n1": 0}, "p1")
	report := calibrate(t, fixtures, positive)

	if len(report.Sweep) != 1 {
		t.Fatalf("sweep has %d entries, want 1", len(report.Sweep))
	}
	if report.Selected.Threshold != 0 || report.Selected.TP != 1 || report.Selected.FP != 1 {
		t.Fatalf("selected = %+v, want threshold 0 with TP 1 FP 1", report.Selected)
	}
}

// A negative score sits below the boundary; the boundary candidate is the
// lowest sweep value, so it admits the fixture instead of reporting F1 0
// from a band midpoint that misses it.
func TestSelectThreshold_NegativeScoreBoundary(t *testing.T) {
	fixtures, positive := scoredFixtures(map[string]float64{"p1": -0.2, "n1": 0.5}, "p1")
	report := calibrate(t, fixtures, positive)

	if report.Selected.Threshold != -0.2 || report.Selected.TP != 1 || report.Selected.FP != 1 {
		t.Fatalf("selected = %+v, want the boundary threshold -0.2 with TP 1 FP 1", report.Selected)
	}
}

func parseTestSet(t *testing.T, manifest string) error {
	t.Helper()
	_, err := parseSet([]byte(manifest), []config.EmbeddingRule{{Name: testRule}})
	return err
}

// A loaded rule with no reviewed positive must not reach calibration: it
// would score against an empty positive set and look perfectly separable.
func TestParseSet_RequiresAPositiveForEveryRule(t *testing.T) {
	manifest := []byte(`{"positives": [{"image_file": "a.png", "signal_name": "rule"}], "negatives": ["b.png"]}`)
	rules := []config.EmbeddingRule{{Name: testRule}, {Name: "uncovered"}}
	if _, err := parseSet(manifest, rules); err == nil {
		t.Fatal("manifest accepted although rule \"uncovered\" has no positive")
	}
	if _, err := parseSet(manifest, rules[:1]); err != nil {
		t.Fatalf("manifest rejected although every rule is covered: %v", err)
	}
}

// The manifest is explicit: every fixture carries a reviewed label, a path
// cannot carry two labels, and an exclusion must say why.
func TestParseSet_AcceptsReviewedManifest(t *testing.T) {
	err := parseTestSet(t, `{
		"positives": [{"image_file": "a.png", "signal_name": "rule"}],
		"negatives": ["b.jpg", "c.jpeg"],
		"excluded": [{"image_file": "d.png", "reason": "ambiguous"}]
	}`)
	if err != nil {
		t.Fatalf("valid manifest rejected: %v", err)
	}
}

func TestParseSet_RejectsUnreviewedShapes(t *testing.T) {
	cases := map[string]string{
		"no negatives":        `{"positives": [{"image_file": "a.png", "signal_name": "rule"}], "negatives": []}`,
		"no positives":        `{"positives": [], "negatives": ["b.png"]}`,
		"unknown rule":        `{"positives": [{"image_file": "a.png", "signal_name": "other"}], "negatives": ["b.png"]}`,
		"positive twice":      `{"positives": [{"image_file": "a.png", "signal_name": "rule"}, {"image_file": "a.png", "signal_name": "rule"}], "negatives": ["b.png"]}`,
		"positive+negative":   `{"positives": [{"image_file": "a.png", "signal_name": "rule"}], "negatives": ["a.png"]}`,
		"negative twice":      `{"positives": [{"image_file": "a.png", "signal_name": "rule"}], "negatives": ["b.png", "b.png"]}`,
		"negative+excluded":   `{"positives": [{"image_file": "a.png", "signal_name": "rule"}], "negatives": ["b.png"], "excluded": [{"image_file": "b.png", "reason": "x"}]}`,
		"exclusion no reason": `{"positives": [{"image_file": "a.png", "signal_name": "rule"}], "negatives": ["b.png"], "excluded": [{"image_file": "c.png"}]}`,
		"bad extension":       `{"positives": [{"image_file": "a.png", "signal_name": "rule"}], "negatives": ["b.gif"]}`,
		"empty path":          `{"positives": [{"image_file": "", "signal_name": "rule"}], "negatives": ["b.png"]}`,
		"legacy roots":        `{"negative_roots": ["images"], "positives": [{"image_file": "a.png", "signal_name": "rule"}]}`,
	}
	for name, manifest := range cases {
		t.Run(name, func(t *testing.T) {
			if err := parseTestSet(t, manifest); err == nil {
				t.Fatalf("manifest accepted: %s", manifest)
			}
		})
	}
}

// A rule the classifier did not score must fail the run, not become a 0 that
// reads as a confident negative; the same goes for duplicate, unknown, and
// non-finite scores.
func TestCollectScores_RequiresOneFiniteScorePerRule(t *testing.T) {
	rules := []config.EmbeddingRule{{Name: "a"}, {Name: "b"}}
	score := func(name string, value float64) classification.EmbeddingRuleScore {
		return classification.EmbeddingRuleScore{Name: name, Score: value}
	}
	got, err := collectScores(rules, []classification.EmbeddingRuleScore{score("a", 0.4), score("b", 0.1)})
	if err != nil || got["a"] != 0.4 || got["b"] != 0.1 {
		t.Fatalf("complete result rejected: %v %v", got, err)
	}
	cases := map[string][]classification.EmbeddingRuleScore{
		"missing rule":   {score("a", 0.4)},
		"duplicate rule": {score("a", 0.4), score("a", 0.5), score("b", 0.1)},
		"unknown rule":   {score("a", 0.4), score("b", 0.1), score("c", 0.2)},
		"nan":            {score("a", math.NaN()), score("b", 0.1)},
		"inf":            {score("a", math.Inf(1)), score("b", 0.1)},
	}
	for name, scored := range cases {
		t.Run(name, func(t *testing.T) {
			if _, err := collectScores(rules, scored); err == nil {
				t.Fatalf("accepted %v", scored)
			}
		})
	}
}

// Only image-modality rules with candidates can be scored; anything else the
// classifier skips, and the calibration must refuse it up front.
func TestValidateRules_RejectsUnscorableRules(t *testing.T) {
	good := config.EmbeddingRule{Name: "img", QueryModality: config.QueryModalityImage, Candidates: []string{"a photo"}}
	if err := validateRules([]config.EmbeddingRule{good}); err != nil {
		t.Fatalf("image rule rejected: %v", err)
	}
	cases := map[string]config.EmbeddingRule{
		"text modality":    {Name: "txt", QueryModality: config.QueryModalityText, Candidates: []string{"a"}},
		"default modality": {Name: "def", Candidates: []string{"a"}},
		"no candidates":    {Name: "empty", QueryModality: config.QueryModalityImage},
	}
	for name, rule := range cases {
		t.Run(name, func(t *testing.T) {
			if err := validateRules([]config.EmbeddingRule{good, rule}); err == nil {
				t.Fatalf("rule %+v accepted", rule)
			}
		})
	}
}

// A band that straddles zero must not be split by a synthetic 0 in the
// sweep: (-0.25, 0.25] (width 0.50) ties (0.40, 0.775] (width 0.375) on F1
// and must win on width, giving midpoint 0.
func TestSelectThreshold_BandStraddlingZeroKeepsItsWidth(t *testing.T) {
	fixtures, positive := scoredFixtures(map[string]float64{
		"n1": -0.25, "p1": 0.25, "n2": 0.30, "n3": 0.40, "p2": 0.775, "p3": 0.80, "n4": 0.85,
	}, "p1", "p2", "p3")
	report := calibrate(t, fixtures, positive)

	if report.Selected.Threshold != 0 || report.Selected.TP != 3 || report.Selected.FP != 3 || report.Selected.FN != 0 {
		t.Fatalf("selected = %+v, want threshold 0 with TP 3 FP 3 FN 0", report.Selected)
	}
	for _, s := range report.Sweep {
		if s.Threshold == 0 {
			t.Fatal("sweep contains a synthetic 0 although no fixture scored 0")
		}
	}
}

func TestCheckCanonical(t *testing.T) {
	for _, ok := range []string{"a.png", "dir/sub/a.jpg", "dir/x.png"} {
		if err := checkCanonical(ok); err != nil {
			t.Errorf("%q rejected: %v", ok, err)
		}
	}
	for _, bad := range []string{"", ".", "/abs/a.png", "../a.png", "dir/../a.png", "dir/..", "./a.png", "dir//a.png", "dir/", "a\\b.png"} {
		if err := checkCanonical(bad); err == nil {
			t.Errorf("%q accepted", bad)
		}
	}
}

func gitRepo(t *testing.T) string {
	t.Helper()
	root := t.TempDir()
	run := func(args ...string) {
		t.Helper()
		cmd := exec.Command("git", append([]string{"-C", root}, args...)...) //nolint:gosec // test helper; fixed git subcommands
		cmd.Env = append(os.Environ(), "GIT_AUTHOR_NAME=t", "GIT_AUTHOR_EMAIL=t@t", "GIT_COMMITTER_NAME=t", "GIT_COMMITTER_EMAIL=t@t")
		if out, err := cmd.CombinedOutput(); err != nil {
			t.Fatalf("git %v: %v\n%s", args, err, out)
		}
	}
	write := func(name, content string) {
		t.Helper()
		if err := os.MkdirAll(filepath.Dir(filepath.Join(root, name)), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(root, name), []byte(content), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	run("init", "-q")
	write("img/tracked.png", "png")
	write(".gitignore", "ignored/\n")
	run("add", ".")
	run("commit", "-q", "-m", "fixtures")
	write("img/untracked.png", "png")
	write("ignored/model.png", "png")
	if err := os.Symlink("tracked.png", filepath.Join(root, "img", "link.png")); err != nil {
		t.Skipf("symlinks unavailable: %v", err)
	}
	return root
}

// Only canonical paths naming tracked regular files of HEAD are inputs the
// recorded commit can reproduce.
func TestBindToCommit_RequiresTrackedRegularFiles(t *testing.T) {
	root := gitRepo(t)
	if err := bindToCommit(root, []string{"img/tracked.png", ".gitignore"}); err != nil {
		t.Fatalf("tracked inputs rejected: %v", err)
	}
	outside := filepath.Join(t.TempDir(), "outside.png")
	if err := os.WriteFile(outside, []byte("png"), 0o644); err != nil {
		t.Fatal(err)
	}
	escape, err := filepath.Rel(root, outside)
	if err != nil {
		t.Fatal(err)
	}
	for name, path := range map[string]string{
		"untracked":     "img/untracked.png",
		"ignored":       "ignored/model.png",
		"symlink":       "img/link.png",
		"missing":       "img/missing.png",
		"directory":     "img",
		"escapes root":  filepath.ToSlash(escape),
		"non-canonical": "img/../img/tracked.png",
	} {
		t.Run(name, func(t *testing.T) {
			if err := bindToCommit(root, []string{"img/tracked.png", path}); err == nil {
				t.Fatalf("%q accepted as a commit-bound input", path)
			}
		})
	}
}

// The manifest and rules paths are resolved through symlinks before they
// are validated and read, so "alias/../rules.yaml" cannot validate as an
// inside file while the OS reads one outside the checkout.
func TestResolveInput_ResolvesSymlinksBeforeCleaning(t *testing.T) {
	root := gitRepo(t)
	cwd, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	outside := t.TempDir()
	if err := os.WriteFile(filepath.Join(outside, "rules.yaml"), []byte("outside"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(filepath.Dir(outside), "rules.yaml"), []byte("outside-parent"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(outside, filepath.Join(root, "alias")); err != nil {
		t.Skipf("symlinks unavailable: %v", err)
	}
	// An inside rules.yaml exists so that lexically cleaning "alias/../rules.yaml"
	// to "rules.yaml" would wrongly succeed; only symlink-first resolution
	// sees that the path leaves the checkout.
	if err := os.WriteFile(filepath.Join(root, "rules.yaml"), []byte("inside"), 0o644); err != nil {
		t.Fatal(err)
	}
	realPath, rel, err := resolveInput(root, filepath.Join(root, "img", "tracked.png"))
	if err != nil || rel != "img/tracked.png" {
		t.Fatalf("inside input: real=%q rel=%q err=%v", realPath, rel, err)
	}
	if got, _ := os.ReadFile(realPath); string(got) != "png" {
		t.Fatalf("resolved path %q reads %q, want the tracked file", realPath, got)
	}
	// The same inputs given relative to the working directory must resolve
	// identically, and a relative "alias/../rules.yaml" must still be rejected.
	if err := os.Chdir(root); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = os.Chdir(cwd) })
	if _, rel, err := resolveInput(".", "img/tracked.png"); err != nil || rel != "img/tracked.png" {
		t.Fatalf("relative inside input: rel=%q err=%v", rel, err)
	}
	if _, rel, err := resolveInput(".", "alias/../rules.yaml"); err == nil {
		t.Fatalf("relative alias/../rules.yaml accepted as %q", rel)
	}
	// A symlink that stays inside the checkout resolves to the file that is
	// actually read, and that tracked path is what gets validated.
	if realPath, rel, err := resolveInput(root, filepath.Join(root, "img", "link.png")); err != nil || rel != "img/tracked.png" || filepath.Base(realPath) != "tracked.png" {
		t.Fatalf("inside symlink: real=%q rel=%q err=%v", realPath, rel, err)
	}
	for name, path := range map[string]string{
		// Built with string concatenation on purpose: filepath.Join would
		// drop "alias/.." lexically and never exercise the symlink.
		"through symlink":       root + "/alias/rules.yaml",
		"cleaned past symlink":  root + "/alias/../rules.yaml",
		"absolute outside root": filepath.Join(outside, "rules.yaml"),
	} {
		t.Run(name, func(t *testing.T) {
			if realPath, rel, err := resolveInput(root, path); err == nil {
				t.Fatalf("%q accepted as %q (%s)", path, rel, realPath)
			}
		})
	}
}
