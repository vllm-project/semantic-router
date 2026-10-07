package classification

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
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/serving"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
)

// fittedModel is the model_sha256 the test artifact was fitted on.
var fittedModel = strings.Repeat("ab", 32)

// servedCategoryModel is a category backend whose runtime serves one model.
type servedCategoryModel struct {
	countingCoreClassifierInitializer
	identity string
}

func (m *servedCategoryModel) ModelSHA256() (string, error) { return m.identity, nil }

// calibratedDomainClassifier builds a classifier whose category backend is
// served the model with the given identity.
func calibratedDomainClassifier(t *testing.T, served string) *Classifier {
	t.Helper()
	dir := t.TempDir()
	artifact, err := json.Marshal(map[string]any{
		"artifact_schema_version": "signal-calibration-artifact/v1",
		"artifact_id":             "sha256:domain-test",
		"status":                  "calibrated",
		"family":                  "domain",
		"scale":                   "label_correctness/v1",
		"model": map[string]any{
			"labels":       []string{"health", "law"},
			"model_sha256": fittedModel,
		},
		"mapping": map[string]any{"knots": [][2]float64{{0.5, 0.2}, {1.0, 0.9}}},
	})
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(dir, "calibration.json")
	if err := os.WriteFile(path, artifact, 0o644); err != nil {
		t.Fatal(err)
	}
	digest := sha256.Sum256(artifact)
	cfg := &config.RouterConfig{}
	cfg.CategoryModel = config.CategoryModel{
		ModelID:             dir,
		CategoryMappingPath: filepath.Join(dir, "category_mapping.json"),
		Threshold:           0.5,
		Calibration:         &config.ScoreCalibrationReference{Path: path, SHA256: hex.EncodeToString(digest[:])},
	}
	cfg.Strategy = config.RoutingStrategyConfidence
	cfg.Decisions = []config.Decision{
		{Name: "law", Priority: 100, Rules: config.RuleNode{Type: config.SignalTypeDomain, Name: "law"}},
		{Name: "health", Priority: 200, Rules: config.RuleNode{Type: config.SignalTypeDomain, Name: "health"}},
	}
	return &Classifier{
		Config: cfg,
		CategoryMapping: &CategoryMapping{
			CategoryToIdx: map[string]int{"health": 0, "law": 1},
			IdxToCategory: map[string]string{"0": "health", "1": "law"},
		},
		categoryInitializer: &servedCategoryModel{identity: served},
	}
}

func TestDomainCalibrationReachesDecisionRanking(t *testing.T) {
	classifier := calibratedDomainClassifier(t, fittedModel)
	if err := classifier.InitializeRuntime(); err != nil {
		t.Fatalf("InitializeRuntime() error = %v", err)
	}
	signals := &SignalResults{
		MatchedDomainRules: []string{"law", "health"},
		SignalConfidences:  map[string]float64{"domain:law": 0.95, "domain:health": 0.6},
	}
	result, err := classifier.EvaluateDecisionWithEngine(signals)
	if err != nil || result == nil {
		t.Fatalf("EvaluateDecisionWithEngine() = %v, %v", result, err)
	}
	ranking := signals.Diagnostics.Ranking
	if result.Decision.Name != "law" || math.Abs(result.Confidence-0.83) > 1e-9 ||
		ranking.ScoreKind != string(config.ScoreKindCalibrated) || ranking.ScoreArtifact != "sha256:domain-test" {
		t.Fatalf("winner %s at %v with ranking %+v, want law at 0.83 naming sha256:domain-test", result.Decision.Name, result.Confidence, ranking)
	}
}

func TestDomainCalibrationFittedOnAnotherModelStopsStartup(t *testing.T) {
	classifier := calibratedDomainClassifier(t, strings.Repeat("cd", 32))
	err := classifier.InitializeRuntime()
	if err == nil || !strings.Contains(err.Error(), "different model") {
		t.Fatalf("InitializeRuntime() error = %v, want the stale calibration refused", err)
	}
}

// A category backend that cannot name its served model, such as a remote
// classify endpoint, never loads a calibration.
func TestDomainCalibrationNeedsTheServedModel(t *testing.T) {
	classifier := calibratedDomainClassifier(t, fittedModel)
	classifier.categoryInitializer = &countingCoreClassifierInitializer{}
	err := classifier.InitializeRuntime()
	if err == nil || !strings.Contains(err.Error(), "served by the model runtime") {
		t.Fatalf("InitializeRuntime() error = %v, want the calibration refused", err)
	}
}

// cardServices serves one deployment's card; no other call is expected.
type cardServices struct {
	serving.Services
	deployment string
	card       modelservice.ModelCard
}

func (s cardServices) Card(_ context.Context, deployment string) (modelservice.ModelCard, error) {
	if deployment != s.deployment {
		return modelservice.ModelCard{}, os.ErrNotExist
	}
	return s.card, nil
}

func TestCategoryBackendReportsTheServedModelIdentity(t *testing.T) {
	spec := config.ResolvedModelBinding{Binding: config.ModelBinding{Deployment: "domain"}}
	services := cardServices{deployment: "domain", card: modelservice.ModelCard{ModelSHA256: fittedModel}}
	backend := ownedCategoryBackend{&ownedSequenceBackend{runtime: serving.New(services, nil), spec: spec}}
	identity, err := backend.ModelSHA256()
	if err != nil || identity != fittedModel {
		t.Fatalf("ModelSHA256() = %q, %v, want %q", identity, err, fittedModel)
	}
	if _, err := (ownedCategoryBackend{&ownedSequenceBackend{runtime: serving.New(nil, nil), spec: spec}}).ModelSHA256(); err == nil {
		t.Fatal("ModelSHA256() without runtime services succeeded")
	}
}
