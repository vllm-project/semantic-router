package decision

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

var calibrationLabels = []string{"health", "law"}

func sha256Hex(data []byte) string {
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:])
}

// writeCalibration writes a model directory and an artifact fitted on it, then
// lets edit change the artifact before it is pinned.
func writeCalibration(t *testing.T, edit func(map[string]any)) (config.ScoreCalibrationReference, string) {
	t.Helper()
	dir := t.TempDir()
	modelDir := filepath.Join(dir, "model")
	if err := os.MkdirAll(modelDir, 0o755); err != nil {
		t.Fatal(err)
	}
	weights := []byte("weights")
	if err := os.WriteFile(filepath.Join(modelDir, "model.safetensors"), weights, 0o644); err != nil {
		t.Fatal(err)
	}
	artifact := map[string]any{
		"artifact_schema_version": "signal-calibration-artifact/v1",
		"artifact_id":             "sha256:test",
		"status":                  "calibrated",
		"family":                  "domain",
		"scale":                   "label_correctness/v1",
		"model": map[string]any{
			"labels": calibrationLabels,
			"files":  map[string]string{"model.safetensors": sha256Hex(weights)},
		},
		"mapping": map[string]any{"knots": [][2]float64{{0.5, 0.2}, {1.0, 0.9}}},
	}
	if edit != nil {
		edit(artifact)
	}
	data, err := json.Marshal(artifact)
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(dir, "calibration.json")
	if err := os.WriteFile(path, data, 0o644); err != nil {
		t.Fatal(err)
	}
	return config.ScoreCalibrationReference{Path: path, SHA256: sha256Hex(data)}, modelDir
}

func TestLoadScoreCalibrationAppliesKnots(t *testing.T) {
	ref, modelDir := writeCalibration(t, nil)
	calibration, err := LoadScoreCalibration(ref, config.SignalTypeDomain, modelDir, calibrationLabels)
	if err != nil {
		t.Fatalf("LoadScoreCalibration() error = %v", err)
	}
	for score, want := range map[float64]float64{0.3: 0.2, 0.5: 0.2, 0.75: 0.55, 1.0: 0.9} {
		if got := calibration.Apply(score); math.Abs(got-want) > 1e-12 {
			t.Fatalf("Apply(%v) = %v, want %v", score, got, want)
		}
	}
	if calibration.ArtifactID != "sha256:test" {
		t.Fatalf("ArtifactID = %q", calibration.ArtifactID)
	}
}

// Every way an artifact can stop describing the running model refuses it, so a
// stale or missing calibration never ranks.
func TestLoadScoreCalibrationRefusesStaleArtifacts(t *testing.T) {
	cases := map[string]struct {
		edit   func(map[string]any)
		mutate func(config.ScoreCalibrationReference, string) (config.ScoreCalibrationReference, string)
		labels []string
		want   string
	}{
		"missing artifact": {
			mutate: func(ref config.ScoreCalibrationReference, dir string) (config.ScoreCalibrationReference, string) {
				ref.Path += ".missing"
				return ref, dir
			},
			want: "no such file",
		},
		"pinned digest differs": {
			mutate: func(ref config.ScoreCalibrationReference, dir string) (config.ScoreCalibrationReference, string) {
				ref.SHA256 = strings.Repeat("0", 64)
				return ref, dir
			},
			want: "pinned sha256",
		},
		"model weights changed": {
			mutate: func(ref config.ScoreCalibrationReference, dir string) (config.ScoreCalibrationReference, string) {
				if err := os.WriteFile(filepath.Join(dir, "model.safetensors"), []byte("retrained"), 0o644); err != nil {
					panic(err)
				}
				return ref, dir
			},
			want: "different model",
		},
		"label order differs": {labels: []string{"law", "health"}, want: "labels"},
		"held-out brier did not improve": {
			edit: func(a map[string]any) { a["status"] = "no_improvement" },
			want: "not calibrated",
		},
		"other scale": {
			edit: func(a map[string]any) { a["scale"] = "other/v1" },
			want: "label_correctness/v1",
		},
		"knots fall": {
			edit: func(a map[string]any) { a["mapping"] = map[string]any{"knots": [][2]float64{{0.5, 0.9}, {1.0, 0.2}}} },
			want: "does not rise",
		},
		"weights unbound": {
			edit: func(a map[string]any) {
				a["model"] = map[string]any{"labels": calibrationLabels, "files": map[string]string{}}
			},
			want: "model.safetensors",
		},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			ref, modelDir := writeCalibration(t, tc.edit)
			if tc.mutate != nil {
				ref, modelDir = tc.mutate(ref, modelDir)
			}
			labels := calibrationLabels
			if tc.labels != nil {
				labels = tc.labels
			}
			_, err := LoadScoreCalibration(ref, config.SignalTypeDomain, modelDir, labels)
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("LoadScoreCalibration() error = %v, want it to mention %q", err, tc.want)
			}
		})
	}
}

func calibratedRanking(t *testing.T, families []string, calibration *ScoreCalibration, decisions []config.Decision, signals *SignalMatches) (*DecisionResult, *RankingTrace) {
	t.Helper()
	engine := NewDecisionEngine(nil, nil, nil, decisions, config.RoutingStrategyConfidence).WithScoreCalibration(families, calibration)
	result, diagnostics, err := engine.EvaluateDecisionsWithDiagnostics(signals)
	if err != nil || result == nil {
		t.Fatalf("EvaluateDecisionsWithDiagnostics() = %v, %v", result, err)
	}
	return result, diagnostics.Ranking
}

func TestCalibratedDomainScoresRankAndNameTheirArtifact(t *testing.T) {
	ref, modelDir := writeCalibration(t, nil)
	calibration, err := LoadScoreCalibration(ref, config.SignalTypeDomain, modelDir, calibrationLabels)
	if err != nil {
		t.Fatal(err)
	}
	law := config.Decision{Name: "law", Priority: 100, Rules: config.RuleNode{Type: "domain", Name: "law"}}
	health := config.Decision{Name: "health", Priority: 200, Rules: config.RuleNode{Type: "domain", Name: "health"}}
	safety := config.Decision{Name: "safety", Priority: 150, Rules: config.RuleNode{Type: "safety", Name: "unsafe"}}
	signals := &SignalMatches{
		DomainRules: []string{"law", "health"},
		SafetyRules: []string{"unsafe"},
		SignalConfidences: map[string]float64{
			"domain:law": 0.95, "domain:health": 0.6, "safety:unsafe": 0.99,
		},
	}
	families := []string{config.SignalTypeDomain}

	winner, trace := calibratedRanking(t, families, calibration, []config.Decision{law, health}, signals)
	if winner.Decision.Name != "law" || math.Abs(winner.Confidence-0.83) > 1e-9 {
		t.Fatalf("winner = %s at %v, want law at the calibrated 0.83", winner.Decision.Name, winner.Confidence)
	}
	if !trace.Comparable || trace.ScoreKind != "calibrated" || trace.ScoreArtifact != "sha256:test" {
		t.Fatalf("trace = %+v, want a comparable calibrated pool naming sha256:test", trace)
	}

	// An uncalibrated probability is a different quantity, so the pool keeps
	// the priority fallback until a second family shares the scale.
	winner, trace = calibratedRanking(t, families, calibration, []config.Decision{law, safety}, signals)
	if winner.Decision.Name != "safety" || trace.Comparable || !strings.Contains(trace.Fallback, "calibrated") {
		t.Fatalf("winner = %s, trace = %+v, want the priority fallback over mixed kinds", winner.Decision.Name, trace)
	}
}

// A family the configuration declares calibrated but whose artifact never
// loaded reports no comparable score instead of its raw probability.
func TestDeclaredCalibrationWithoutArtifactNeverRanks(t *testing.T) {
	law := config.Decision{Name: "law", Priority: 100, Rules: config.RuleNode{Type: "domain", Name: "law"}}
	health := config.Decision{Name: "health", Priority: 200, Rules: config.RuleNode{Type: "domain", Name: "health"}}
	signals := &SignalMatches{
		DomainRules:       []string{"law", "health"},
		SignalConfidences: map[string]float64{"domain:law": 0.95, "domain:health": 0.6},
	}
	winner, trace := calibratedRanking(t, []string{config.SignalTypeDomain}, nil, []config.Decision{law, health}, signals)
	if winner.Decision.Name != "health" || trace.Comparable || trace.ScoreArtifact != "" {
		t.Fatalf("winner = %s, trace = %+v, want the priority fallback and no artifact", winner.Decision.Name, trace)
	}
}
