package classification

import (
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

// calibratedDomainClassifier builds a classifier whose domain model directory
// holds the weights a calibration artifact was fitted on.
func calibratedDomainClassifier(t *testing.T, weights string) *Classifier {
	t.Helper()
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "model.safetensors"), []byte(weights), 0o644); err != nil {
		t.Fatal(err)
	}
	fitted := sha256.Sum256([]byte("weights"))
	artifact, err := json.Marshal(map[string]any{
		"artifact_schema_version": "signal-calibration-artifact/v1",
		"artifact_id":             "sha256:domain-test",
		"status":                  "calibrated",
		"family":                  "domain",
		"scale":                   "label_correctness/v1",
		"model": map[string]any{
			"labels": []string{"health", "law"},
			"files":  map[string]string{"model.safetensors": hex.EncodeToString(fitted[:])},
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
		categoryInitializer: &countingCoreClassifierInitializer{},
	}
}

func TestDomainCalibrationReachesDecisionRanking(t *testing.T) {
	classifier := calibratedDomainClassifier(t, "weights")
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

func TestDomainCalibrationFittedOnOtherWeightsStopsStartup(t *testing.T) {
	classifier := calibratedDomainClassifier(t, "retrained weights")
	err := classifier.InitializeRuntime()
	if err == nil || !strings.Contains(err.Error(), "different model") {
		t.Fatalf("InitializeRuntime() error = %v, want the stale calibration refused", err)
	}
}
