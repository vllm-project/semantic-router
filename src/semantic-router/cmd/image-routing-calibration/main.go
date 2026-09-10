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
// The labelled calibration set lives in testdata/calibration-set.json: the
// positive labels (image, rule, the candidate phrase it depicts) plus the
// directories whose PNG/JPEG files form the negative pool. Everything not
// named as a positive is a negative for every rule. The set is versioned by
// the repository commit: the report records that commit, flags a dirty tree,
// and lists every resolved fixture with its sha256, so two reports are
// comparable without duplicating any image.
//
// The generated report is PR evidence and is not committed; regenerate it with the
// invocation below whenever the model artifact, the fixture set, or a rule's
// candidate list changes, and mirror any threshold change into the
// multimodal-routing E2E IntelligentRoute CRD
// (TestImageRoutingPack_MatchesMultimodalE2EProfile enforces the lockstep).
//
// -check turns the run into a gate (exit 2 unless every shipped threshold
// equals the report-selected value). The manually dispatched workflow
// .github/workflows/image-routing-calibration.yml runs it that way at a
// chosen ref and uploads the reports, so reviewers get exact-head evidence.
//
// Known state at snapshot fdf8e01b7b0f3a69ac1ac8e2a64dcb1ede177ba4:
// identifier_document_imagery (0.61) and ambient_office_imagery (0.54) are
// separable with ~0.10 and ~0.14 cosine headroom on each side.
// code_or_terminal_imagery is NOT separable on repo imagery — dark UI
// screenshots outscore several genuine code/terminal images — so its shipped
// 0.4632 is the max-F1 band midpoint (F1 0.375) and the E2E code fixture
// (score 0.4748) clears it by only ~0.012. Treat that rule as the first suspect
// when the multimodal E2E profile regresses; a stronger code fixture is the
// real fix.
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

// calibrationSet is the labelled input. Every PNG/JPEG under NegativeRoots at
// the checked-out commit is a fixture; a fixture is positive for a rule only
// when named in Positives and is a negative for every other rule. The set is
// versioned by the repository commit rather than by per-file hashes: git
// already content-addresses the assets, and the report records the commit,
// whether the tree was dirty, and every resolved fixture's sha256.
type calibrationSet struct {
	NegativeRoots []string        `json:"negative_roots"`
	Positives     []positiveLabel `json:"positives"`
}

type positiveLabel struct {
	ImageFile   string `json:"image_file"`
	SignalName  string `json:"signal_name"`
	Description string `json:"description,omitempty"`
}

// fixtureExtensions mirrors the image crate features compiled into
// candle-binding (jpeg, png); anything else fails to decode at the FFI.
var fixtureExtensions = map[string]string{".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}

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
	// Source pins the fixture side of the run: the repository commit the
	// images were read from and whether anything under the scanned roots was
	// modified or untracked at the time (a dirty run is not reproducible).
	Source struct {
		Commit string `json:"repo_commit"`
		Dirty  bool   `json:"repo_dirty"`
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
	flag.Parse()
	if *modelPath == "" || *artifactRevision == "" || *casesPath == "" {
		fatal("-model (or MULTIMODAL_MODEL_PATH), -artifact-revision, and -cases are required")
	}

	rules := loadRules(*rulesPath)
	set := loadSet(*casesPath, rules)
	fixtures := enumerateFixtures(*fixtureRoot, set)
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
	report.Source.Commit, report.Source.Dirty = repoState(*fixtureRoot, set.NegativeRoots)
	if report.Source.Dirty {
		fmt.Fprintln(os.Stderr, "WARNING: uncommitted changes under the fixture roots; this report is not reproducible from its commit")
	}
	for _, fixture := range fixtures {
		report.Fixtures = append(report.Fixtures, scoreFixture(classifier, *fixtureRoot, fixture, set, rules))
	}
	for _, rule := range rules {
		report.Rules = append(report.Rules, calibrateRule(rule, report.Fixtures))
	}

	writeReports(report, *output, *markdown)
	if *check {
		os.Exit(checkShippedThresholds(report, *artifactRevision, *expectRevision))
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
func checkShippedThresholds(report calibrationReport, gotRevision, wantRevision string) int {
	code := 0
	fmt.Println("image-routing calibration gate")
	fmt.Printf("  fixtures=%d repo_commit=%s dirty=%t artifact_revision=%s\n",
		len(report.Fixtures), report.Source.Commit, report.Source.Dirty, gotRevision)
	if wantRevision != "" && wantRevision != gotRevision {
		fmt.Printf("  WARNING: model snapshot %s differs from calibrated snapshot %s; rerun the calibration and refresh the docs\n",
			gotRevision, wantRevision)
	}
	if report.Source.Dirty {
		fmt.Println("  WARNING: uncommitted changes under the fixture roots; this run is not reproducible from its commit")
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
	var set calibrationSet
	if err := json.Unmarshal(data, &set); err != nil {
		fatal("parse calibration set: %v", err)
	}
	if len(set.NegativeRoots) == 0 || len(set.Positives) == 0 {
		fatal("calibration set needs at least one negative_roots entry and one positive")
	}
	known := map[string]bool{}
	for _, rule := range rules {
		known[rule.Name] = true
	}
	for _, label := range set.Positives {
		if !known[label.SignalName] {
			fatal("positive %q names unknown rule %q", label.ImageFile, label.SignalName)
		}
		if _, ok := fixtureExtensions[strings.ToLower(filepath.Ext(label.ImageFile))]; !ok {
			fatal("positive %q: unsupported image extension (need png/jpg/jpeg)", label.ImageFile)
		}
	}
	return set
}

// enumerateFixtures walks every negative root for decodable images and unions
// in the positive paths, returning repo-relative paths in sorted order. A
// positive that does not exist on disk is fatal so a renamed asset cannot
// silently drop a label.
func enumerateFixtures(root string, set calibrationSet) []string {
	seen := map[string]bool{}
	for _, dir := range set.NegativeRoots {
		err := filepath.WalkDir(filepath.Join(root, dir), func(path string, entry os.DirEntry, err error) error {
			if err != nil {
				return err
			}
			if entry.IsDir() {
				return nil
			}
			if _, ok := fixtureExtensions[strings.ToLower(filepath.Ext(path))]; !ok {
				return nil
			}
			rel, err := filepath.Rel(root, path)
			if err != nil {
				return err
			}
			seen[filepath.ToSlash(rel)] = true
			return nil
		})
		if err != nil {
			fatal("scan negative root %q: %v", dir, err)
		}
	}
	for _, label := range set.Positives {
		if _, err := os.Stat(filepath.Join(root, label.ImageFile)); err != nil {
			fatal("positive fixture %q: %v", label.ImageFile, err)
		}
		seen[label.ImageFile] = true
	}
	fixtures := make([]string, 0, len(seen))
	for path := range seen {
		fixtures = append(fixtures, path)
	}
	sort.Strings(fixtures)
	return fixtures
}

// repoState returns the HEAD commit of the checkout at root and whether any
// path under roots is modified or untracked. Fixtures are versioned by this
// commit, so the report must say when the tree did not match it.
func repoState(root string, roots []string) (string, bool) {
	head, err := exec.Command("git", "-C", root, "rev-parse", "HEAD").Output()
	if err != nil {
		fatal("resolve repository commit for fixtures (is %q a git checkout?): %v", root, err)
	}
	args := append([]string{"-C", root, "status", "--porcelain", "--"}, roots...)
	status, err := exec.Command("git", args...).Output()
	if err != nil {
		fatal("check fixture roots for uncommitted changes: %v", err)
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
	report := fixtureReport{Path: path, SHA256: fileSHA(resolved), Scores: map[string]float64{}}
	for _, label := range set.Positives {
		if label.ImageFile == path && !contains(report.PositiveFor, label.SignalName) {
			report.PositiveFor = append(report.PositiveFor, label.SignalName)
		}
	}
	for _, rule := range rules {
		report.Scores[rule.Name] = 0
	}
	for _, score := range result.Scores {
		report.Scores[score.Name] = score.Score
	}
	return report
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
	// The sweep is built only from fixture scores (plus 0), never from the
	// threshold currently shipped, so the recommendation cannot depend on the
	// value being calibrated. The shipped value is evaluated separately below.
	values := map[float64]bool{0: true}
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
	// Non-separable: score every band between consecutive fixture scores at its
	// midpoint. Ties on F1 go to the wider band, measured by that band's own
	// width (not the global margin, which is dominated by the overlap and can
	// prefer a narrow band). A remaining tie keeps the lower band.
	var best thresholdResult
	bestWidth := -1.0
	for i := 1; i < len(sweep); i++ {
		// Band (sweep[i-1].Threshold, sweep[i].Threshold]: same matrix as sweep[i].
		width := sweep[i].Threshold - sweep[i-1].Threshold
		candidate := evaluateThreshold(sweep[i-1].Threshold+width/2, rule, positive, fixtures)
		if bestWidth < 0 || candidate.F1 > best.F1 || candidate.F1 == best.F1 && width > bestWidth {
			best, bestWidth = candidate, width
		}
	}
	if rounded := evaluateThreshold(roundTwoPlaces(best.Threshold), rule, positive, fixtures); sameMatrix(rounded, best) {
		return rounded
	}
	return best
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
		dirty = " (**tree was dirty under the fixture roots; not reproducible from this commit**)"
	}
	fmt.Fprintf(&b, "- Fixture source: repository commit `%s`%s\n", report.Source.Commit, dirty)
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
