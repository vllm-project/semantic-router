// Command image-routing-calibration derives the per-rule thresholds shipped in
// config/fragments/signal/embedding/image-routing.yaml from a versioned
// positive/negative image set, and emits a reproducible report.
//
// Issue #2165: the previous thresholds were tuned before the encoder
// correctness fixes in #1927, #1928 and #1943 and no longer described the
// embedding distribution. It wires the REAL candle multimodal FFI, so scores
// match the path the router takes at runtime.
//
// Threshold selection: for each rule, P is the set of fixtures depicting
// something one of that rule's shipped candidate phrases names, N is every
// other fixture. If min(P) > max(N), take the midpoint of that band and record
// the margin (distance to the nearest fixture on either side); otherwise take
// the midpoint of the max-F1 band and record the residual confusion matrix
// rather than presenting a non-separable rule as calibrated. Candidates are an
// INPUT to calibration, never an output — the report pins each candidate list
// by sha256.
//
// The labelled calibration set lives in testdata/calibration-set.json and is
// an explicit, hand-reviewed manifest: positive labels (image, rule, the
// candidate phrase it depicts), the list of reviewed negatives (images that
// depict no candidate phrase of any rule), and excluded images that were
// judged ambiguous, with the reason. Nothing is inferred from directory
// contents, so adding an image to the repository does not silently add an
// unlabelled negative. A positive is a negative for every other rule. The set
// is versioned by the repository commit: the report records that commit,
// flags any uncommitted change in the worktree (fixtures, manifest, rules,
// or code), and lists every listed and excluded fixture with its sha256, so
// two reports are comparable without duplicating any image.
//
// The generated report is PR evidence and is not committed; regenerate it with the
// invocation below whenever the model artifact, the fixture set, or a rule's
// candidate list changes, and mirror any threshold change into the
// multimodal-routing E2E IntelligentRoute CRD
// (TestImageRoutingPack_MatchesMultimodalE2EProfile enforces the lockstep).
//
// -check turns the run into a gate (exit 2 unless every shipped threshold
// equals the report-selected value; with -require-clean also unless the
// worktree matches the recorded commit). The workflow
// .github/workflows/image-routing-calibration.yml runs it that way at the
// exact pull-request head and uploads the reports, so reviewers get
// reproducible evidence.
//
// Known state at snapshot fdf8e01b7b0f3a69ac1ac8e2a64dcb1ede177ba4:
// identifier_document_imagery (0.61) and ambient_office_imagery (0.54) are
// separable with ~0.10 and ~0.14 cosine headroom on each side.
// code_or_terminal_imagery is NOT separable on repo imagery — dark UI
// screenshots outscore several genuine code/terminal images — so its shipped
// 0.47 is the max-F1 band midpoint (F1 0.444, 8 FP / 7 FN over 13 positives)
// and the E2E code fixture (score 0.4748) clears it by only ~0.005. Treat that
// rule as the first suspect when the multimodal E2E profile regresses; a
// stronger code fixture is the real fix.
//
// Scores are the classifier's prototype blend under aggregation_method=max
// (best_weight*best + (1-best_weight)*mean(top_m), defaults 0.75 / 2), not a
// raw max cosine; a deployment that overrides prototype_scoring shifts every
// threshold.
//
// Build/run (needs the candle lib + the multimodal model):
//
//	hf download llm-semantic-router/multi-modal-embed-small \
//	  config.json model.safetensors tokenizer.json tokenizer_config.json \
//	  special_tokens_map.json \
//	  --revision fdf8e01b7b0f3a69ac1ac8e2a64dcb1ede177ba4 \
//	  --local-dir models/mom-embedding-multimodal
//
//	cd src/semantic-router && \
//	  DYLD_LIBRARY_PATH=$PWD/../../candle-binding/target/release \
//	  LD_LIBRARY_PATH=$PWD/../../candle-binding/target/release \
//	  go run ./cmd/image-routing-calibration \
//	    -model ../../models/mom-embedding-multimodal \
//	    -artifact-revision fdf8e01b7b0f3a69ac1ac8e2a64dcb1ede177ba4 \
//	    -rules ../../config/fragments/signal/embedding/image-routing.yaml \
//	    -cases ./cmd/image-routing-calibration/testdata/calibration-set.json \
//	    -fixture-root ../.. \
//	    -output /tmp/image-routing-calibration.json \
//	    -markdown /tmp/image-routing-calibration.md
//
// -artifact-revision is recorded verbatim: pass the snapshot the model was
// downloaded at (the --revision above; the router's own downloader tracks
// "main", see pkg/modeldownload/config_parser.go), or the report claims
// reproducibility it does not have. The CI workflow pins the download to the
// same snapshot and fails closed if the resolved revision differs.
package main

import (
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"

	"gopkg.in/yaml.v2"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

const modelRepository = "llm-semantic-router/multi-modal-embed-small"

// calibrationSet is the labelled input: every fixture is listed explicitly
// with the label a reviewer gave it. A fixture is positive for a rule only
// when named in Positives and is a negative for every other rule; Negatives
// depict no candidate phrase of any rule; Excluded images are ambiguous and
// are not scored, but must still exist so the manifest cannot go stale. The
// set is versioned by the repository commit rather than by per-file hashes:
// git already content-addresses the assets, and the report records the
// commit, whether the tree was dirty, and every listed fixture's sha256.
type calibrationSet struct {
	Positives []positiveLabel `json:"positives"`
	Negatives []string        `json:"negatives"`
	Excluded  []excludedLabel `json:"excluded,omitempty"`
}

type positiveLabel struct {
	ImageFile   string `json:"image_file"`
	SignalName  string `json:"signal_name"`
	Description string `json:"description,omitempty"`
}

type excludedLabel struct {
	ImageFile string `json:"image_file"`
	Reason    string `json:"reason"`
}

// fixtureExtensions mirrors the image crate features compiled into
// candle-binding (jpeg, png); anything else fails to decode at the FFI.
var fixtureExtensions = map[string]string{".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}

type excludedFixture struct {
	Path   string `json:"path"`
	SHA256 string `json:"sha256"`
	Reason string `json:"reason"`
}

type fixtureReport struct {
	Path        string             `json:"path"`
	SHA256      string             `json:"sha256"`
	PositiveFor []string           `json:"positive_for,omitempty"`
	Scores      map[string]float64 `json:"scores"`
}

type ruleReport struct {
	Name            string            `json:"name"`
	CandidateSHA256 string            `json:"candidate_sha256"`
	Positives       int               `json:"positive_fixtures"`
	Negatives       int               `json:"negative_fixtures"`
	Shipped         thresholdResult   `json:"shipped"`
	Selected        thresholdResult   `json:"selected"`
	Sweep           []thresholdResult `json:"sweep"`
}

type thresholdResult struct {
	Threshold float64 `json:"threshold"`
	TP        int     `json:"true_positive"`
	FP        int     `json:"false_positive"`
	TN        int     `json:"true_negative"`
	FN        int     `json:"false_negative"`
	Precision float64 `json:"precision"`
	Recall    float64 `json:"recall"`
	F1        float64 `json:"f1"`
	// Margin is the distance from Threshold to the nearest fixture score on
	// either side (negative when a positive sits below or a negative sits
	// above the threshold). The midpoint of a separable band maximizes it.
	Margin    float64 `json:"margin"`
	Separable bool    `json:"separable"`
	// HeadroomPositive is min(positive) - Threshold; HeadroomNegative is
	// Threshold - max(negative). Both are absolute cosine gaps, which is the
	// quantity encoder drift eats into.
	HeadroomPositive float64 `json:"headroom_positive"`
	HeadroomNegative float64 `json:"headroom_negative"`
}

type calibrationReport struct {
	Model struct {
		Repository      string `json:"repository"`
		ArtifactSHA     string `json:"artifact_revision"`
		TargetDimension int    `json:"target_dimension"`
		ModelType       string `json:"model_type"`
		Aggregation     string `json:"aggregation_method"`
		// Scoring records the effective prototype-scoring parameters. With
		// aggregation_method=max the runtime score is
		// best_weight*best + (1-best_weight)*mean(top_m), not a raw max
		// cosine, so thresholds shift if a deployment overrides these.
		Scoring config.PrototypeScoringConfig `json:"prototype_scoring"`
	} `json:"model"`
	// Source pins the repository side of the run: the commit the fixtures,
	// manifest, rules, and tool were read from, whether anything in the
	// worktree was modified or untracked at the time (a dirty run is not
	// reproducible from the commit; only this run's own report files are
	// ignored), and the images the calibration set excludes as ambiguous,
	// hashed so a later reviewer can tell exactly which asset was excluded.
	Source struct {
		Commit   string            `json:"repo_commit"`
		Dirty    bool              `json:"repo_dirty"`
		Excluded []excludedFixture `json:"excluded_fixtures,omitempty"`
	} `json:"source"`
	Fixtures []fixtureReport `json:"fixtures"`
	Rules    []ruleReport    `json:"rules"`
}

func main() {
	modelPath := flag.String("model", os.Getenv("MULTIMODAL_MODEL_PATH"), "multimodal model directory")
	rulesPath := flag.String("rules", "../../config/fragments/signal/embedding/image-routing.yaml", "image-routing YAML fragment")
	casesPath := flag.String("cases", "./cmd/image-routing-calibration/testdata/calibration-set.json", "labelled calibration set JSON")
	fixtureRoot := flag.String("fixture-root", ".", "directory that relative fixture paths in the cases file resolve against")
	artifactRevision := flag.String("artifact-revision", "", "resolved model snapshot commit (required for reproducible reports)")
	output := flag.String("output", "image-routing-calibration.json", "JSON report path")
	markdown := flag.String("markdown", "image-routing-calibration.md", "Markdown report path")
	check := flag.Bool("check", false, "gate mode: exit 2 unless every rule's report-selected threshold equals the shipped value")
	expectRevision := flag.String("expect-artifact-revision", "", "snapshot the shipped thresholds were calibrated on; a different -artifact-revision is reported as a warning, not a failure")
	requireClean := flag.Bool("require-clean", false, "with -check: fail unless the worktree matches the recorded commit, so the report is reproducible evidence")
	flag.Parse()
	if *modelPath == "" || *artifactRevision == "" || *casesPath == "" {
		fatal("-model (or MULTIMODAL_MODEL_PATH), -artifact-revision, and -cases are required")
	}

	rules := loadRules(*rulesPath)
	if err := validateRules(rules); err != nil {
		fatal("rules: %v", err)
	}
	set := loadSet(*casesPath, rules)
	fixtures := enumerateFixtures(set)
	// Every input the report attributes to the commit must actually come from
	// it: the fixtures, the excluded assets, the manifest, and the rules.
	inputs := append([]string{}, fixtures...)
	for _, label := range set.Excluded {
		inputs = append(inputs, label.ImageFile)
	}
	inputs = append(inputs, repoRelative(*fixtureRoot, *casesPath), repoRelative(*fixtureRoot, *rulesPath))
	if err := bindToCommit(*fixtureRoot, inputs); err != nil {
		fatal("%v", err)
	}
	if err := candle_binding.InitMultiModalEmbeddingModel(*modelPath, true); err != nil {
		fatal("initialize multimodal model: %v", err)
	}
	classifier, err := classification.NewEmbeddingClassifier(rules, config.HNSWConfig{
		ModelType: "multimodal", TargetDimension: 384, PreloadEmbeddings: true,
	})
	if err != nil {
		fatal("initialize classifier: %v", err)
	}

	report := calibrationReport{}
	report.Model.Repository = modelRepository
	report.Model.ArtifactSHA = *artifactRevision
	report.Model.TargetDimension = 384
	report.Model.ModelType = "multimodal"
	report.Model.Aggregation = "max"
	report.Model.Scoring = config.PrototypeScoringConfig{}.WithDefaults()
	report.Source.Commit, report.Source.Dirty = repoState(*fixtureRoot, *output, *markdown)
	for _, label := range set.Excluded {
		report.Source.Excluded = append(report.Source.Excluded, excludedFixture{
			Path: label.ImageFile, SHA256: fileSHA(filepath.Join(*fixtureRoot, label.ImageFile)), Reason: label.Reason,
		})
	}
	if report.Source.Dirty {
		fmt.Fprintln(os.Stderr, "WARNING: worktree has uncommitted changes; this report is not reproducible from its commit")
	}
	for _, fixture := range fixtures {
		report.Fixtures = append(report.Fixtures, scoreFixture(classifier, *fixtureRoot, fixture, set, rules))
	}
	for _, rule := range rules {
		report.Rules = append(report.Rules, calibrateRule(rule, report.Fixtures))
	}

	writeReports(report, *output, *markdown)
	if *check {
		os.Exit(checkShippedThresholds(report, *artifactRevision, *expectRevision, *requireClean))
	}
}

func writeReports(report calibrationReport, output, markdown string) {
	data, err := json.MarshalIndent(report, "", "  ")
	if err != nil {
		fatal("marshal report: %v", err)
	}
	if err := os.WriteFile(output, append(data, '\n'), 0o644); err != nil {
		fatal("write JSON report: %v", err)
	}
	if err := os.WriteFile(markdown, []byte(renderMarkdown(report)), 0o644); err != nil {
		fatal("write Markdown report: %v", err)
	}
}

// shippedThresholdTolerance absorbs float32 storage of the YAML value and the
// tool's two-decimal rounding of a separable midpoint; anything larger means
// the shipped value is not the report's pick.
const shippedThresholdTolerance = 5e-5

// checkShippedThresholds is the trust gate behind -check: the checked-in
// thresholds must be exactly what this run selects, so a CI run of this
// command is reproducible evidence rather than a contributor's local claim.
// Returns the process exit code (0 pass, 2 mismatch). A model snapshot that
// differs from the one the thresholds were calibrated on is a warning: the
// calibration should be rerun, but the gate only asserts what it can verify.
// A dirty worktree is a warning locally and, with requireClean, a failure:
// CI evidence must be reproducible from the commit it names.
func checkShippedThresholds(report calibrationReport, gotRevision, wantRevision string, requireClean bool) int {
	code := 0
	fmt.Println("image-routing calibration gate")
	fmt.Printf("  fixtures=%d repo_commit=%s dirty=%t artifact_revision=%s\n",
		len(report.Fixtures), report.Source.Commit, report.Source.Dirty, gotRevision)
	if wantRevision != "" && wantRevision != gotRevision {
		fmt.Printf("  WARNING: model snapshot %s differs from calibrated snapshot %s; rerun the calibration and refresh the docs\n",
			gotRevision, wantRevision)
	}
	if report.Source.Dirty {
		if requireClean {
			fmt.Println("  FAIL: worktree has uncommitted changes; this run is not reproducible from its commit")
			code = 2
		} else {
			fmt.Println("  WARNING: worktree has uncommitted changes; this run is not reproducible from its commit")
		}
	}
	for _, rule := range report.Rules {
		shipped, selected := rule.Shipped.Threshold, rule.Selected.Threshold
		status := "ok"
		if diff := shipped - selected; diff > shippedThresholdTolerance || diff < -shippedThresholdTolerance {
			status = "MISMATCH"
			code = 2
		}
		fmt.Printf("  %-32s shipped=%.4f selected=%.4f separable=%t F1=%.3f FP=%d FN=%d  %s\n",
			rule.Name, shipped, selected, rule.Selected.Separable, rule.Selected.F1, rule.Selected.FP, rule.Selected.FN, status)
	}
	if code != 0 {
		fmt.Println("  FAIL: a shipped threshold is not the report-selected value; rerun the calibration and ship what it selects")
	} else {
		fmt.Println("  PASS: every shipped threshold equals the report-selected value")
	}
	return code
}

func loadRules(path string) []config.EmbeddingRule {
	data, err := os.ReadFile(path)
	if err != nil {
		fatal("read rules: %v", err)
	}
	var document config.CanonicalConfig
	if err := yaml.Unmarshal(data, &document); err != nil {
		fatal("parse rules: %v", err)
	}
	if len(document.Routing.Signals.Embeddings) == 0 {
		fatal("rules file contains no embeddings")
	}
	return document.Routing.Signals.Embeddings
}

func loadSet(path string, rules []config.EmbeddingRule) calibrationSet {
	data, err := os.ReadFile(path)
	if err != nil {
		fatal("read calibration set: %v", err)
	}
	set, err := parseSet(data, rules)
	if err != nil {
		fatal("calibration set %s: %v", path, err)
	}
	return set
}

// parseSet decodes and validates a manifest: at least one negative, at least
// one reviewed positive for every loaded rule (a rule with no positive would
// otherwise calibrate against an empty set and look perfectly separable),
// every positive naming a known rule, every path decodable by the FFI, and
// no path under two labels (a positive may be listed once per rule).
func parseSet(data []byte, rules []config.EmbeddingRule) (calibrationSet, error) {
	var set calibrationSet
	if err := json.Unmarshal(data, &set); err != nil {
		return set, fmt.Errorf("parse: %w", err)
	}
	if len(set.Positives) == 0 || len(set.Negatives) == 0 {
		return set, fmt.Errorf("needs at least one positive and one negative")
	}
	known := map[string]bool{}
	for _, rule := range rules {
		known[rule.Name] = true
	}
	label := map[string]string{}
	claim := func(path, kind string) error {
		if path == "" {
			return fmt.Errorf("%s entry has an empty image_file", kind)
		}
		if _, ok := fixtureExtensions[strings.ToLower(filepath.Ext(path))]; !ok {
			return fmt.Errorf("%s %q: unsupported image extension (need png/jpg/jpeg)", kind, path)
		}
		if previous, seen := label[path]; seen && previous != kind {
			return fmt.Errorf("%q is listed as both %s and %s", path, previous, kind)
		}
		label[path] = kind
		return nil
	}
	seenPositive := map[string]bool{}
	covered := map[string]bool{}
	for _, entry := range set.Positives {
		if !known[entry.SignalName] {
			return set, fmt.Errorf("positive %q names unknown rule %q", entry.ImageFile, entry.SignalName)
		}
		key := entry.ImageFile + "\x00" + entry.SignalName
		if seenPositive[key] {
			return set, fmt.Errorf("positive %q is listed twice for rule %q", entry.ImageFile, entry.SignalName)
		}
		seenPositive[key] = true
		covered[entry.SignalName] = true
		if err := claim(entry.ImageFile, "positive"); err != nil {
			return set, err
		}
	}
	for _, rule := range rules {
		if !covered[rule.Name] {
			return set, fmt.Errorf("rule %q has no reviewed positive; every loaded rule needs at least one", rule.Name)
		}
	}
	for _, path := range set.Negatives {
		if label[path] == "negative" {
			return set, fmt.Errorf("negative %q is listed twice", path)
		}
		if err := claim(path, "negative"); err != nil {
			return set, err
		}
	}
	for _, entry := range set.Excluded {
		if entry.Reason == "" {
			return set, fmt.Errorf("excluded %q needs a reason", entry.ImageFile)
		}
		if label[entry.ImageFile] == "excluded" {
			return set, fmt.Errorf("excluded %q is listed twice", entry.ImageFile)
		}
		if err := claim(entry.ImageFile, "excluded"); err != nil {
			return set, err
		}
	}
	return set, nil
}

// enumerateFixtures returns the scored fixtures (positives and negatives) as
// repo-relative paths in sorted order. Existence and commit membership are
// checked by bindToCommit, together with the excluded assets.
func enumerateFixtures(set calibrationSet) []string {
	seen := map[string]bool{}
	for _, label := range set.Positives {
		seen[label.ImageFile] = true
	}
	for _, path := range set.Negatives {
		seen[path] = true
	}
	fixtures := make([]string, 0, len(seen))
	for path := range seen {
		fixtures = append(fixtures, path)
	}
	sort.Strings(fixtures)
	return fixtures
}

// repoRelative converts a command-line path into the repo-relative form the
// manifest uses, so the manifest and rules files can be bound to the commit
// like every fixture.
func repoRelative(root, path string) string {
	rel, err := filepath.Rel(absolutePath(root), absolutePath(path))
	if err != nil {
		fatal("resolve %q against %q: %v", path, root, err)
	}
	return filepath.ToSlash(rel)
}

// bindToCommit checks that every input path is a canonical repo-relative
// path naming a tracked regular file of the checkout at root, so the report
// can attribute it to the recorded commit. A path that escapes the root,
// goes through a symlink, or is merely present on disk (untracked or
// ignored) is rejected: git status would not report such an input as dirty,
// and the commit could not reproduce it.
func bindToCommit(root string, paths []string) error {
	rootReal, err := filepath.EvalSymlinks(absolutePath(root))
	if err != nil {
		return fmt.Errorf("resolve repository root %q: %v", root, err)
	}
	for _, path := range paths {
		if err := checkCanonical(path); err != nil {
			return err
		}
		real, err := filepath.EvalSymlinks(filepath.Join(rootReal, filepath.FromSlash(path)))
		if err != nil {
			return fmt.Errorf("input %q: %v", path, err)
		}
		if rel, err := filepath.Rel(rootReal, real); err != nil || filepath.ToSlash(rel) != path {
			return fmt.Errorf("input %q resolves outside the repository or through a symlink (%s)", path, real)
		}
		info, err := os.Lstat(real)
		if err != nil {
			return fmt.Errorf("input %q: %v", path, err)
		}
		if !info.Mode().IsRegular() {
			return fmt.Errorf("input %q is not a regular file", path)
		}
	}
	tracked, err := trackedFiles(rootReal, paths)
	if err != nil {
		return err
	}
	for _, path := range paths {
		if mode, ok := tracked[path]; !ok {
			return fmt.Errorf("input %q is not tracked at the recorded commit (untracked or ignored files cannot be reproduced from it)", path)
		} else if mode != "100644" && mode != "100755" {
			return fmt.Errorf("input %q is tracked as mode %s, not a regular file", path, mode)
		}
	}
	return nil
}

// checkCanonical accepts only clean, relative, forward-slash paths that stay
// beneath the root, so one file has one spelling in the manifest.
func checkCanonical(path string) error {
	switch {
	case path == "" || path == ".":
		return fmt.Errorf("input path is empty")
	case filepath.IsAbs(path) || strings.HasPrefix(path, "/"):
		return fmt.Errorf("input %q must be relative to the repository root", path)
	case strings.Contains(path, "\\"):
		return fmt.Errorf("input %q must use forward slashes", path)
	case path == ".." || strings.HasPrefix(path, "../") || strings.Contains(path, "/../") || strings.HasSuffix(path, "/.."):
		return fmt.Errorf("input %q escapes the repository root", path)
	case filepath.ToSlash(filepath.Clean(filepath.FromSlash(path))) != path:
		return fmt.Errorf("input %q is not in canonical form (want %q)", path, filepath.ToSlash(filepath.Clean(filepath.FromSlash(path))))
	}
	return nil
}

// trackedFiles returns the git mode of every requested path that HEAD
// tracks, keyed by repo-relative path.
func trackedFiles(root string, paths []string) (map[string]string, error) {
	args := append([]string{"-C", root, "ls-tree", "-z", "HEAD", "--"}, paths...)
	out, err := exec.Command("git", args...).Output()
	if err != nil {
		return nil, fmt.Errorf("list tracked inputs at HEAD: %v", err)
	}
	tracked := map[string]string{}
	for _, entry := range strings.Split(string(out), "\x00") {
		if entry == "" {
			continue
		}
		// "<mode> <type> <hash>\t<path>"
		meta, path, found := strings.Cut(entry, "\t")
		fields := strings.Fields(meta)
		if !found || len(fields) != 3 {
			return nil, fmt.Errorf("unexpected ls-tree entry %q", entry)
		}
		tracked[path] = fields[0]
	}
	return tracked, nil
}

// repoState returns the HEAD commit of the checkout at root and whether the
// worktree differs from it anywhere: fixtures, manifest, rule YAML, the
// scoring code, and this tool are all inputs the commit is supposed to pin.
// Only this run's own report files are excluded when they live inside the
// checkout (the CI workflow writes them outside it).
func repoState(root string, outputs ...string) (string, bool) {
	head, err := exec.Command("git", "-C", root, "rev-parse", "HEAD").Output()
	if err != nil {
		fatal("resolve repository commit (is %q a git checkout?): %v", root, err)
	}
	args := []string{"-C", root, "status", "--porcelain", "--", "."}
	for _, output := range outputs {
		if rel, relErr := filepath.Rel(absolutePath(root), absolutePath(output)); relErr == nil && !strings.HasPrefix(rel, "..") {
			args = append(args, ":(exclude)"+filepath.ToSlash(rel))
		}
	}
	status, err := exec.Command("git", args...).Output()
	if err != nil {
		fatal("check worktree for uncommitted changes: %v", err)
	}
	return strings.TrimSpace(string(head)), len(strings.TrimSpace(string(status))) > 0
}

func scoreFixture(classifier *classification.EmbeddingClassifier, root, path string, set calibrationSet, rules []config.EmbeddingRule) fixtureReport {
	resolved := path
	if !filepath.IsAbs(resolved) {
		resolved = filepath.Join(root, path)
	}
	result, err := classifier.ClassifyDetailedMultimodal(config.QueryModalityImage, imageDataURI(resolved))
	if err != nil {
		fatal("score fixture %q: %v", resolved, err)
	}
	report := fixtureReport{Path: path, SHA256: fileSHA(resolved)}
	for _, label := range set.Positives {
		if label.ImageFile == path && !contains(report.PositiveFor, label.SignalName) {
			report.PositiveFor = append(report.PositiveFor, label.SignalName)
		}
	}
	scores, err := collectScores(rules, result.Scores)
	if err != nil {
		fatal("score fixture %q: %v", path, err)
	}
	report.Scores = scores
	return report
}

// validateRules refuses a rule set the calibration cannot score honestly:
// every rule must be an image-modality rule with candidates, otherwise the
// classifier silently skips it and the tool would have nothing to calibrate.
func validateRules(rules []config.EmbeddingRule) error {
	for _, rule := range rules {
		if modality := rule.EffectiveQueryModality(); modality != config.QueryModalityImage {
			return fmt.Errorf("rule %q has query_modality %q; only image rules can be calibrated here", rule.Name, modality)
		}
		if len(rule.Candidates) == 0 {
			return fmt.Errorf("rule %q has no candidates", rule.Name)
		}
	}
	return nil
}

// collectScores turns the classifier's result into one finite score per
// loaded rule. A rule the classifier did not score (empty prototype bank,
// skipped modality) must fail the run rather than count as 0, which would
// look like a confident negative; duplicates and unknown names are rejected
// for the same reason.
func collectScores(rules []config.EmbeddingRule, scored []classification.EmbeddingRuleScore) (map[string]float64, error) {
	known := map[string]bool{}
	for _, rule := range rules {
		known[rule.Name] = true
	}
	scores := make(map[string]float64, len(rules))
	for _, score := range scored {
		if !known[score.Name] {
			return nil, fmt.Errorf("classifier returned a score for unknown rule %q", score.Name)
		}
		if _, dup := scores[score.Name]; dup {
			return nil, fmt.Errorf("classifier returned two scores for rule %q", score.Name)
		}
		if math.IsNaN(score.Score) || math.IsInf(score.Score, 0) {
			return nil, fmt.Errorf("rule %q scored a non-finite value %v", score.Name, score.Score)
		}
		scores[score.Name] = score.Score
	}
	for _, rule := range rules {
		if _, ok := scores[rule.Name]; !ok {
			return nil, fmt.Errorf("classifier returned no score for rule %q", rule.Name)
		}
	}
	return scores, nil
}

func calibrateRule(rule config.EmbeddingRule, fixtures []fixtureReport) ruleReport {
	positive := map[string]bool{}
	for _, fixture := range fixtures {
		for _, name := range fixture.PositiveFor {
			if name == rule.Name {
				positive[fixture.Path] = true
			}
		}
	}
	// The sweep is built only from the distinct fixture scores, never from the
	// threshold currently shipped (so the recommendation cannot depend on the
	// value being calibrated) and never from a synthetic value such as 0
	// (which would split a band that straddles it and skew the width
	// tie-break). The shipped value is evaluated separately below.
	values := map[float64]bool{}
	for _, fixture := range fixtures {
		values[fixture.Scores[rule.Name]] = true
	}
	thresholds := make([]float64, 0, len(values))
	for threshold := range values {
		thresholds = append(thresholds, threshold)
	}
	sort.Float64s(thresholds)
	report := ruleReport{Name: rule.Name, CandidateSHA256: hashCandidates(rule.Candidates)}
	for _, threshold := range thresholds {
		report.Sweep = append(report.Sweep, evaluateThreshold(threshold, rule.Name, positive, fixtures))
	}
	report.Selected = selectThreshold(rule.Name, positive, fixtures, report.Sweep)
	// Metrics at the threshold the pack currently ships, so the report always
	// validates the checked-in value rather than only the algorithmic pick.
	report.Shipped = evaluateThreshold(float64(rule.SimilarityThreshold), rule.Name, positive, fixtures)
	report.Positives = len(positive)
	report.Negatives = len(fixtures) - len(positive)
	return report
}

func evaluateThreshold(threshold float64, rule string, positive map[string]bool, fixtures []fixtureReport) thresholdResult {
	result := thresholdResult{Threshold: threshold}
	for _, fixture := range fixtures {
		result.count(fixture.Scores[rule] >= threshold, positive[fixture.Path])
	}
	result.Precision = ratio(result.TP, result.TP+result.FP)
	result.Recall = ratio(result.TP, result.TP+result.FN)
	if result.Precision+result.Recall > 0 {
		result.F1 = 2 * result.Precision * result.Recall / (result.Precision + result.Recall)
	}
	minPositive, maxNegative := positiveNegativeBounds(rule, positive, fixtures)
	result.HeadroomPositive = minPositive - threshold
	result.HeadroomNegative = threshold - maxNegative
	result.Margin = min(result.HeadroomPositive, result.HeadroomNegative)
	result.Separable = result.FP == 0 && result.FN == 0
	return result
}

// count adds one fixture to the confusion matrix.
func (r *thresholdResult) count(actual, expected bool) {
	switch {
	case actual && expected:
		r.TP++
	case actual:
		r.FP++
	case expected:
		r.FN++
	default:
		r.TN++
	}
}

// selectThreshold picks the threshold to recommend. Scores are compared with
// >=, so every threshold between two consecutive distinct fixture scores yields
// the same confusion matrix; within a band we therefore prefer the midpoint,
// which maximizes the distance to the nearest fixture on either side.
//
// Separable: the band between max(negative) and min(positive). The midpoint is
// rounded to two decimals when the rounded value still separates perfectly.
// Non-separable: the band with the highest F1 (ties broken toward the wider
// band), reported with its residual confusion matrix.
func selectThreshold(rule string, positive map[string]bool, fixtures []fixtureReport, sweep []thresholdResult) thresholdResult {
	minPositive, maxNegative := positiveNegativeBounds(rule, positive, fixtures)
	if minPositive > maxNegative {
		midpoint := (minPositive + maxNegative) / 2
		rounded := roundTwoPlaces(midpoint)
		if candidate := evaluateThreshold(rounded, rule, positive, fixtures); candidate.Separable {
			return candidate
		}
		return evaluateThreshold(midpoint, rule, positive, fixtures)
	}
	// Non-separable: the lowest sweep value is the boundary candidate (it is
	// the only threshold that admits a fixture scoring exactly at the floor,
	// and the only candidate at all when every fixture shares one score), then
	// every band between consecutive fixture scores at its midpoint. Ties on
	// F1 go to the wider band, measured by that band's own width (the global
	// margin is dominated by the overlap and can prefer a narrow band); the
	// boundary has width zero so it wins only on F1. A remaining tie keeps
	// the lower band.
	best := evaluateThreshold(sweep[0].Threshold, rule, positive, fixtures)
	bestWidth := 0.0
	for i := 1; i < len(sweep); i++ {
		// Band (sweep[i-1].Threshold, sweep[i].Threshold]: same matrix as sweep[i].
		width := sweep[i].Threshold - sweep[i-1].Threshold
		candidate := evaluateThreshold(sweep[i-1].Threshold+width/2, rule, positive, fixtures)
		if better, tie := compareF1(candidate, best); better || tie && width > bestWidth {
			best, bestWidth = candidate, width
		}
	}
	if rounded := evaluateThreshold(roundTwoPlaces(best.Threshold), rule, positive, fixtures); sameMatrix(rounded, best) {
		return rounded
	}
	return best
}

// compareF1 orders two results by F1 computed exactly from the integer
// confusion counts, F1 = 2TP / (2TP + FP + FN), via cross-multiplication.
// The float F1 stored on the result goes through precision and recall and can
// differ in the last bit for mathematically equal values, which would defeat
// the tie-break. A zero denominator (no positives and no predictions) is F1 0.
func compareF1(a, b thresholdResult) (better, tie bool) {
	numA, denA := 2*a.TP, 2*a.TP+a.FP+a.FN
	numB, denB := 2*b.TP, 2*b.TP+b.FP+b.FN
	if denA == 0 {
		numA, denA = 0, 1
	}
	if denB == 0 {
		numB, denB = 0, 1
	}
	left, right := numA*denB, numB*denA
	return left > right, left == right
}

func roundTwoPlaces(value float64) float64 {
	return float64(int(value*100+0.5)) / 100
}

func sameMatrix(a, b thresholdResult) bool {
	return a.TP == b.TP && a.FP == b.FP && a.TN == b.TN && a.FN == b.FN
}

func positiveNegativeBounds(rule string, positive map[string]bool, fixtures []fixtureReport) (float64, float64) {
	minPositive, maxNegative := 2.0, -2.0
	for _, fixture := range fixtures {
		if positive[fixture.Path] && fixture.Scores[rule] < minPositive {
			minPositive = fixture.Scores[rule]
		}
		if !positive[fixture.Path] && fixture.Scores[rule] > maxNegative {
			maxNegative = fixture.Scores[rule]
		}
	}
	return minPositive, maxNegative
}

func renderMarkdown(report calibrationReport) string {
	var b strings.Builder
	b.WriteString("# Image Routing Calibration Report\n\n")
	b.WriteString("Generated by `cmd/image-routing-calibration`. Regenerate with the invocation in\n")
	b.WriteString("the package doc comment of `cmd/image-routing-calibration/main.go`.\n\n")
	fmt.Fprintf(&b, "- Model repository: `%s`\n- Artifact revision: `%s`\n- Target dimension: `%d`\n- Model type: `%s`\n- Aggregation: `%s`\n- Calibration fixtures: `%d`\n",
		report.Model.Repository, report.Model.ArtifactSHA, report.Model.TargetDimension,
		report.Model.ModelType, report.Model.Aggregation, len(report.Fixtures))
	dirty := ""
	if report.Source.Dirty {
		dirty = " (**worktree had uncommitted changes; not reproducible from this commit**)"
	}
	fmt.Fprintf(&b, "- Source: repository commit `%s`%s\n", report.Source.Commit, dirty)
	for _, excluded := range report.Source.Excluded {
		fmt.Fprintf(&b, "- Excluded as ambiguous (not scored): `%s` (`%s`): %s\n", excluded.Path, excluded.SHA256, excluded.Reason)
	}
	fmt.Fprintf(&b, "- Effective score: `%.2f*best + %.2f*mean(top %d)` (prototype_scoring defaults; "+
		"deployments that override `best_weight`/`top_m` shift every threshold)\n\n",
		report.Model.Scoring.BestWeight, 1-report.Model.Scoring.BestWeight, report.Model.Scoring.TopM)

	b.WriteString("## Shipped thresholds\n\n")
	b.WriteString("Metrics are computed at the threshold currently in `image-routing.yaml`.\n\n")
	b.WriteString("Headroom is the absolute cosine gap from the threshold to the nearest positive\n")
	b.WriteString("(`+pos`) and nearest negative (`+neg`); a negative value means a misclassified fixture.\n\n")
	b.WriteString("| Rule | Shipped | Pos | Neg | TP | FP | FN | Precision | Recall | F1 | Headroom +pos | Headroom +neg | Separable |\n")
	b.WriteString("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|\n")
	for _, rule := range report.Rules {
		s := rule.Shipped
		fmt.Fprintf(&b, "| `%s` | `%.4f` | %d | %d | %d | %d | %d | %.3f | %.3f | %.3f | `%+.4f` | `%+.4f` | %t |\n",
			rule.Name, s.Threshold, rule.Positives, rule.Negatives,
			s.TP, s.FP, s.FN, s.Precision, s.Recall, s.F1, s.HeadroomPositive, s.HeadroomNegative, s.Separable)
	}

	b.WriteString("\n## Best achievable threshold per rule\n\n")
	b.WriteString("Selection rule: if `min(positive) > max(negative)` take the midpoint of that band;\n")
	b.WriteString("otherwise take the midpoint of the max-F1 band and record the residual confusion\n")
	b.WriteString("matrix. Margin is the distance to the nearest fixture on either side.\n\n")
	b.WriteString("| Rule | Best | F1 | FP | FN | Margin | Separable |\n|---|---:|---:|---:|---:|---:|:---:|\n")
	for _, rule := range report.Rules {
		s := rule.Selected
		fmt.Fprintf(&b, "| `%s` | `%.4f` | `%.3f` | %d | %d | `%+.4f` | `%t` |\n",
			rule.Name, s.Threshold, s.F1, s.FP, s.FN, s.Margin, s.Separable)
	}

	b.WriteString("\n## Candidate list digests\n\n")
	b.WriteString("A change to any rule's candidates invalidates this report.\n\n")
	b.WriteString("| Rule | Candidates |\n|---|---|\n")
	for _, rule := range report.Rules {
		fmt.Fprintf(&b, "| `%s` | `%s` |\n", rule.Name, rule.CandidateSHA256)
	}
	return b.String()
}

// imageDataURI reads an image file and renders it as a base64 data URI.
// ClassifyDetailedMultimodal resolves payloads through
// MultiModalEncodeImageFromBase64, which does not accept filesystem paths
// despite the doc comment on that method.
func imageDataURI(path string) string {
	data, err := os.ReadFile(path)
	if err != nil {
		fatal("read fixture %q: %v", path, err)
	}
	mediaType, ok := fixtureExtensions[strings.ToLower(filepath.Ext(path))]
	if !ok {
		fatal("fixture %q: unsupported image extension", path)
	}
	return "data:" + mediaType + ";base64," + base64.StdEncoding.EncodeToString(data)
}

func absolutePath(path string) string {
	resolved, err := filepath.Abs(path)
	if err != nil {
		fatal("resolve %q: %v", path, err)
	}
	return resolved
}

func fileSHA(path string) string {
	data, err := os.ReadFile(path)
	if err != nil {
		fatal("read fixture %q: %v", path, err)
	}
	sum := sha256.Sum256(data)
	return "sha256:" + hex.EncodeToString(sum[:])
}

func hashCandidates(candidates []string) string {
	data, _ := json.Marshal(candidates)
	sum := sha256.Sum256(data)
	return "sha256:" + hex.EncodeToString(sum[:])
}

func ratio(numerator, denominator int) float64 {
	if denominator == 0 {
		return 0
	}
	return float64(numerator) / float64(denominator)
}

func contains(items []string, value string) bool {
	for _, item := range items {
		if item == value {
			return true
		}
	}
	return false
}

func fatal(format string, args ...interface{}) {
	fmt.Fprintf(os.Stderr, format+"\n", args...)
	os.Exit(1)
}
