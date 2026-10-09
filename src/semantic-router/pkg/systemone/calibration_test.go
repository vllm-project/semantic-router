package systemone

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func calibrationFixture(t *testing.T) (*NativeRequest, *config.RoutingRecipe, NativeCalibration, Candidate) {
	t.Helper()
	request, err := ParseNativeRequest(json.RawMessage(singleRequest))
	if err != nil {
		t.Fatal(err)
	}
	risk := 0.05
	plan := cascadePlan()
	plan.Quality = &config.NativeQualityConfig{Type: "calibrated", Calibration: "held-out", Loss: "bundle_error", MaxRisk: &risk}
	plan.Stages[0].Accept = nil
	recipe := &config.RoutingRecipe{Name: "native", Profile: config.RoutingProfile{
		Decisions: []config.Decision{{Name: "classify", Algorithm: plan}},
	}}
	identity := InferenceIdentity{ModelID: "fast-actual", Revision: "revision-a", ModelSHA256: strings.Repeat("a", 64), Engine: "candle", Profile: "exact", Numerics: "exact", Accelerator: "cpu"}
	var response map[string]any
	if decodeErr := json.Unmarshal([]byte(uncertainResponse), &response); decodeErr != nil {
		t.Fatal(decodeErr)
	}
	response["meta"] = identity
	body, err := json.Marshal(response)
	if err != nil {
		t.Fatal(err)
	}
	candidate := Candidate{Stage: "fast", Model: "kai", Body: body, Invocation: body, History: []StageEvidence{{Stage: "fast", Model: "kai", Outcome: "response", Response: body}}}
	taskSHA, err := CalibrationTaskSHA256(request)
	if err != nil {
		t.Fatal(err)
	}
	policySHA, err := CalibrationPolicySHA256(recipe, &recipe.Profile.Decisions[0])
	if err != nil {
		t.Fatal(err)
	}
	artifact := NativeCalibration{
		SchemaVersion: "systemone-calibration/v1", Loss: "bundle_error", Feature: "min_top_probability", PolicySHA256: policySHA, Alpha: 0.05,
		Certification: CalibrationCertification{Population: "fixture task population", SourceManifestSHA256: strings.Repeat("b", 64), Unit: "independent_source_bundle", Sampling: "iid_source_bundles", PolicyFrozenBeforeCollection: true, IndependentFromTraining: true},
		Cells:         []CalibrationCell{{TaskSHA256: taskSHA, History: []CalibrationHistory{{Stage: "fast", Model: "kai", Identity: identity}}, Bins: []CalibrationBin{{Lower: 0.5, Upper: 1, Accept: true, Count: 100, Errors: 0}}}},
	}
	artifact.Certification.FrozenGatesSHA256, err = CalibrationGateSHA256(artifact.Cells)
	if err != nil {
		t.Fatal(err)
	}
	return request, recipe, artifact, candidate
}

func calibrationLoader(t *testing.T, recipe *config.RoutingRecipe, artifact NativeCalibration) (QualityEvaluator, error) {
	t.Helper()
	cfg := calibrationResource(t, artifact)
	return LoadCalibration(cfg, recipe, &recipe.Profile.Decisions[0])
}

func calibrationResource(t *testing.T, artifact NativeCalibration) *config.RouterConfig {
	t.Helper()
	data, err := json.Marshal(artifact)
	if err != nil {
		t.Fatal(err)
	}
	sum := sha256.Sum256(data)
	path := filepath.Join(t.TempDir(), "evidence.json")
	if writeErr := os.WriteFile(path, data, 0o600); writeErr != nil {
		t.Fatal(writeErr)
	}
	cfg := &config.RouterConfig{Evaluation: &config.CanonicalEvaluation{Calibrations: []config.CalibrationArtifact{{Name: "held-out", Source: path, SHA256: hex.EncodeToString(sum[:])}}}}
	return cfg
}

func TestCalibrationRelativeArtifactIsLoadedOnce(t *testing.T) {
	request, recipe, artifact, candidate := calibrationFixture(t)
	cfg := calibrationResource(t, artifact)
	resource := &cfg.Evaluation.Calibrations[0]
	absPath := resource.Source
	cfg.ConfigBaseDir = filepath.Dir(absPath)
	resource.Source = filepath.Base(absPath)
	evaluate, err := LoadCalibration(cfg, recipe, &recipe.Profile.Decisions[0])
	if err != nil {
		t.Fatal(err)
	}
	if writeErr := os.WriteFile(absPath, []byte(`{"mutated":true}`), 0o600); writeErr != nil {
		t.Fatal(writeErr)
	}
	if accepted, evaluationErr := evaluate(request, candidate); evaluationErr != nil || !accepted {
		t.Fatalf("retained evidence changed: accepted=%v error=%v", accepted, evaluationErr)
	}
	if _, err = LoadCalibration(cfg, recipe, &recipe.Profile.Decisions[0]); err == nil {
		t.Fatal("new generation accepted changed artifact bytes")
	}
}

func TestCalibrationLoadsFrozenEvidenceAndRejectsUnknownRequests(t *testing.T) {
	for _, change := range []string{"none", "question", "option", "identity", "missing_identity", "history", "error", "changed_body", "unknown_bin"} {
		t.Run(change, func(t *testing.T) {
			request, recipe, artifact, candidate := calibrationFixture(t)
			evaluate, err := calibrationLoader(t, recipe, artifact)
			if err != nil {
				t.Fatal(err)
			}
			switch change {
			case "question":
				request.Body = json.RawMessage(strings.Replace(string(request.Body), "Type?", "New task?", 1))
			case "option":
				request.Body = json.RawMessage(strings.TrimSuffix(string(request.Body), "}") + `,"options":{"max_tokens":64}}`)
			case "identity", "missing_identity":
				replacement := "revision-b"
				if change == "missing_identity" {
					replacement = ""
				}
				candidate.Body = json.RawMessage(strings.Replace(string(candidate.Body), "revision-a", replacement, 1))
				candidate.History[0].Response = candidate.Body
			case "history":
				candidate.History[0].Stage = "different-arrival"
			case "error":
				candidate.History[0].Outcome = "error"
			case "changed_body":
				candidate.Body = json.RawMessage(certainResponse)
			case "unknown_bin":
				artifact.Cells[0].Bins[0].Lower = 0.8
				artifact.Certification.FrozenGatesSHA256, _ = CalibrationGateSHA256(artifact.Cells)
				evaluate, err = calibrationLoader(t, recipe, artifact)
				if err != nil {
					t.Fatal(err)
				}
			}
			accepted, err := evaluate(request, candidate)
			if err != nil || accepted != (change == "none") {
				t.Fatalf("accepted=%v error=%v", accepted, err)
			}
		})
	}
}

func TestCalibrationRejectsIncompatibleOrAdaptiveEvidence(t *testing.T) {
	for _, change := range []string{"policy", "training", "freeze", "sampling", "manifest", "gate_digest", "adaptive_gate", "duplicate", "overlap", "count", "zero_accept", "identity", "signals", "judge"} {
		t.Run(change, func(t *testing.T) {
			_, recipe, artifact, _ := calibrationFixture(t)
			switch change {
			case "policy":
				recipe.Profile.Decisions[0].Algorithm.Budget.MaxCalls++
			case "training":
				artifact.Certification.IndependentFromTraining = false
			case "freeze":
				artifact.Certification.PolicyFrozenBeforeCollection = false
			case "sampling":
				artifact.Certification.Sampling = "stratified_task_average"
			case "manifest":
				artifact.Certification.SourceManifestSHA256 = "missing"
			case "gate_digest":
				artifact.Cells[0].Bins[0].Lower = 0.9
			case "adaptive_gate":
				artifact.Cells[0].Bins[0].Errors = 20
			case "duplicate":
				artifact.Cells = append(artifact.Cells, artifact.Cells[0])
			case "overlap":
				artifact.Cells[0].Bins = append(artifact.Cells[0].Bins, artifact.Cells[0].Bins[0])
			case "count":
				artifact.Cells[0].Bins[0].Errors = 101
			case "zero_accept":
				artifact.Cells[0].Bins[0].Count = 0
			case "identity":
				artifact.Cells[0].History[0].Identity.ModelSHA256 = "mutable"
			case "signals":
				recipe.Profile.Signals.KeywordRules = []config.KeywordRule{{Name: "gate"}}
			case "judge":
				recipe.Profile.Decisions[0].Algorithm.Stages = append(recipe.Profile.Decisions[0].Algorithm.Stages, config.CascadeStage{Name: "judge", Kind: "judge", Model: "reviewer"})
			}
			if _, err := calibrationLoader(t, recipe, artifact); err == nil {
				t.Fatal("invalid evidence accepted")
			}
		})
	}
}

func TestCalibrationExecutorChecksEveryConditionalArrivalIdentity(t *testing.T) {
	request, recipe, artifact, first := calibrationFixture(t)
	strong := json.RawMessage(strings.ReplaceAll(string(first.Body), "fast-actual", "strong-actual"))
	identity, _ := responseInferenceIdentity(strong)
	firstStep := artifact.Cells[0].History[0]
	artifact.Cells[0].Bins[0].Accept = false
	artifact.Cells[0].Bins[0].Count = 0
	artifact.Cells = append(artifact.Cells, CalibrationCell{
		TaskSHA256: artifact.Cells[0].TaskSHA256,
		History:    []CalibrationHistory{firstStep, {Stage: "strong", Model: "nox", Identity: identity}},
		Bins:       []CalibrationBin{{Lower: 0, Upper: 1, Accept: true, Count: 100, Errors: 0}},
	})
	artifact.Certification.FrozenGatesSHA256, _ = CalibrationGateSHA256(artifact.Cells)
	evaluate, err := calibrationLoader(t, recipe, artifact)
	if err != nil {
		t.Fatal(err)
	}
	executor, err := NewExecutor(recipe.Profile.Decisions[0].Algorithm, evaluate)
	if err != nil {
		t.Fatal(err)
	}
	for _, alteredPriorIdentity := range []bool{false, true} {
		calls := 0
		result, err := executor.Execute(context.Background(), request, func(_ context.Context, model string, _ json.RawMessage) (int, []byte, error) {
			calls++
			if model == "nox" {
				return 200, strong, nil
			}
			body := first.Body
			if alteredPriorIdentity {
				body = json.RawMessage(strings.ReplaceAll(string(body), "revision-a", "mutable-revision"))
			}
			return 200, body, nil
		})
		if alteredPriorIdentity {
			if err == nil {
				t.Fatal("correct final identity hid a changed earlier model")
			}
		} else if err != nil || result.Stage != "strong" || calls != 2 {
			t.Fatalf("result=%v calls=%d error=%v", result, calls, err)
		}
	}
}

func TestCalibrationCountsCannotEnableFrozenRejectedGate(t *testing.T) {
	request, recipe, artifact, candidate := calibrationFixture(t)
	artifact.Cells[0].Bins[0].Accept = false
	artifact.Cells[0].Bins[0].Count = 0 // An unvisited downstream cell has no evidence.
	artifact.Certification.FrozenGatesSHA256, _ = CalibrationGateSHA256(artifact.Cells)
	evaluate, err := calibrationLoader(t, recipe, artifact)
	if err != nil {
		t.Fatal(err)
	}
	if accepted, _ := evaluate(request, candidate); accepted {
		t.Fatal("unvisited cell passed")
	}
	artifact.Cells[0].Bins[0].Count = 1000000
	evaluate, err = calibrationLoader(t, recipe, artifact)
	if err != nil {
		t.Fatal(err)
	}
	if accepted, _ := evaluate(request, candidate); accepted {
		t.Fatal("certification counts enabled a previously rejected gate")
	}
}

func TestCalibrationSimultaneousCorrectionIncludesAllDeclaredBins(t *testing.T) {
	_, recipe, artifact, _ := calibrationFixture(t)
	// n=100, zero errors passes 5% individually, but ten simultaneous cells do
	// not. Do not ignore bins that the current request does not visit.
	for i := 0; i < 9; i++ {
		cell := artifact.Cells[0]
		cell.TaskSHA256 = strings.Repeat(string(rune('0'+i)), 64)
		artifact.Cells = append(artifact.Cells, cell)
	}
	artifact.Certification.FrozenGatesSHA256, _ = CalibrationGateSHA256(artifact.Cells)
	if _, err := calibrationLoader(t, recipe, artifact); err == nil || !strings.Contains(err.Error(), "max_risk") {
		t.Fatalf("uncorrected cells passed: %v", err)
	}
}

func TestCalibrationFingerprintsBindSemanticsWithoutDeploymentGeneration(t *testing.T) {
	request, recipe, _, _ := calibrationFixture(t)
	initial, _ := CalibrationTaskSHA256(request)
	request.Body = json.RawMessage(strings.Replace(string(request.Body), "hello", "a different input", 1))
	changed, _ := CalibrationTaskSHA256(request)
	if changed != initial {
		t.Fatal("task fingerprint bound an individual input")
	}
	before, _ := CalibrationPolicySHA256(recipe, &recipe.Profile.Decisions[0])
	recipe.Description = "new explanatory text"
	recipe.Profile.Decisions[0].Description = "new description"
	recipe.Profile.Decisions[0].Algorithm.Quality.Calibration = "resource-relocated"
	after, _ := CalibrationPolicySHA256(recipe, &recipe.Profile.Decisions[0])
	if before != after {
		t.Fatal("non-execution metadata changed calibration policy")
	}
	recipe.Profile.Decisions[0].Algorithm.Stages[0].Accept = gate(0.95)
	after, _ = CalibrationPolicySHA256(recipe, &recipe.Profile.Decisions[0])
	if before == after {
		t.Fatal("prior arrival gate was not bound")
	}
}

func TestCalibrationTaskFingerprintIgnoresEvidenceVisibility(t *testing.T) {
	request, _, _, _ := calibrationFixture(t)
	expected, err := CalibrationTaskSHA256(request)
	if err != nil {
		t.Fatal(err)
	}
	for _, option := range []string{`null`, `{}`, `{"return_meta":true}`, `{"return_meta":false}`} {
		request.Body = json.RawMessage(strings.TrimSuffix(singleRequest, "}") + `,"options":` + option + `}`)
		actual, hashErr := CalibrationTaskSHA256(request)
		if hashErr != nil || actual != expected {
			t.Fatalf("evidence visibility changed task identity: options=%s hash=%s err=%v", option, actual, hashErr)
		}
	}
	request.Body = json.RawMessage(strings.TrimSuffix(singleRequest, "}") + `,"options":{"return_meta":true,"max_tokens":64}}`)
	actual, err := CalibrationTaskSHA256(request)
	if err != nil || actual == expected {
		t.Fatal("inference-affecting options escaped task binding")
	}
}

func TestCalibrationClopperPearsonUpperBound(t *testing.T) {
	// Interior values independently calculated with scipy.stats.beta.ppf.
	for _, test := range []struct {
		errors, count int
		alpha, want   float64
	}{
		{0, 96, .05, -math.Expm1(math.Log(.05) / 96)},
		{10, 10, .05, 1},
		{1, 10, .05, .39416330243650466},
		{5, 10, .05, .7775588989918706},
		{25, 1000, .005, .04067217464897199},
		{500, 1000, .05, .5264822687643087},
		{999, 1000, .001, .9999989995001669},
		{10, 1000000, .05, .000016962160188281603},
	} {
		got, err := calibrationRiskUpper(test.errors, test.count, test.alpha)
		if err != nil || math.Abs(got-test.want) > 1e-10 {
			t.Fatalf("k=%d n=%d alpha=%g got=%.15g want=%.15g err=%v", test.errors, test.count, test.alpha, got, test.want, err)
		}
	}
	for _, values := range [][3]float64{{0, 0, .05}, {2, 1, .05}, {-1, 10, .05}, {0, 10, 0}, {0, 10, 1}, {0, 10, math.NaN()}} {
		if _, err := calibrationRiskUpper(int(values[0]), int(values[1]), values[2]); err == nil {
			t.Fatalf("invalid statistics accepted: %v", values)
		}
	}
}

func TestCalibrationTaskKeepsModelVisibleCriteriaOrder(t *testing.T) {
	first, err := ParseNativeRequest(json.RawMessage(`{"state":"hello","questions":{"task":{"type":"choice","instructions":"Which?","criteria":{"a":"A","b":"B"}}}}`))
	if err != nil {
		t.Fatal(err)
	}
	second, err := ParseNativeRequest(json.RawMessage(`{"state":"hello","questions":{"task":{"type":"choice","instructions":"Which?","criteria":{"b":"B","a":"A"}}}}`))
	if err != nil {
		t.Fatal(err)
	}
	firstHash, err := CalibrationTaskSHA256(first)
	if err != nil {
		t.Fatal(err)
	}
	secondHash, err := CalibrationTaskSHA256(second)
	if err != nil {
		t.Fatal(err)
	}
	if firstHash == secondHash {
		t.Fatal("reordered model options reused another template's certificate")
	}
	compact, err := ParseNativeRequest(json.RawMessage(`{ "state": "changed", "questions": { "task": { "type": "choice", "instructions": "Which?", "criteria": { "a": "A", "b": "B" } } } }`))
	if err != nil {
		t.Fatal(err)
	}
	compactHash, err := CalibrationTaskSHA256(compact)
	if err != nil || compactHash != firstHash {
		t.Fatalf("formatting changed task hash: %s %s %v", compactHash, firstHash, err)
	}
}
