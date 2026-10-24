package config

import (
	"reflect"
	"strings"
	"testing"
)

func TestValidateCategoryCalibration(t *testing.T) {
	ref := &ScoreCalibrationReference{Path: "calibration.json", SHA256: strings.Repeat("a", 64)}
	for _, testCase := range []struct {
		name  string
		model CategoryModel
		want  string
	}{
		{name: "absent", model: CategoryModel{Threshold: 0.3}},
		{name: "local model at 0.5", model: CategoryModel{Threshold: 0.5, Calibration: ref}},
		{
			name:  "digest not hex",
			model: CategoryModel{Threshold: 0.5, Calibration: &ScoreCalibrationReference{Path: "c.json", SHA256: "abc"}},
			want:  "calibration.sha256",
		},
		{
			name:  "remote backend",
			model: CategoryModel{Threshold: 0.5, Calibration: ref, Backend: &RemoteClassifierBackend{}},
			want:  "cannot be used with a backend",
		},
		{
			name:  "threshold lets a second label match",
			model: CategoryModel{Threshold: 0.3, Calibration: ref},
			want:  "at least 0.5",
		},
	} {
		t.Run(testCase.name, func(t *testing.T) {
			err := validateCategoryCalibration(&testCase.model)
			if testCase.want == "" {
				if err != nil {
					t.Fatalf("validateCategoryCalibration() error = %v", err)
				}
				return
			}
			if err == nil || !strings.Contains(err.Error(), testCase.want) {
				t.Fatalf("validateCategoryCalibration() error = %v, want %q", err, testCase.want)
			}
		})
	}
}

// The load-time warning and the engine read the same declaration, so a pool
// that mixes calibrated domain scores with another probability is reported.
func TestCalibratedDomainIsItsOwnScoreKind(t *testing.T) {
	cfg := &RouterConfig{}
	cfg.CategoryModel.Calibration = &ScoreCalibrationReference{}
	families := cfg.CalibratedScoreFamilies()
	if got := SignalScoreKind(SignalTypeDomain, families...); got != ScoreKindCalibrated {
		t.Fatalf("SignalScoreKind(domain) = %q, want calibrated", got)
	}
	if got := SignalScoreKind(SignalTypeKB, SignalTypeKB); got != ScoreKindSimilarity {
		t.Fatalf("SignalScoreKind(kb) = %q, only a probability can be calibrated", got)
	}
	got := ambiguousConfidencePools([]Decision{
		leafDecision("law", 1, SignalTypeDomain),
		leafDecision("unsafe", 1, SignalTypeSafety),
	}, families...)
	want := []confidencePoolFallback{{
		Pool:      1,
		Decisions: []string{"law", "unsafe"},
		Kinds:     []string{string(ScoreKindCalibrated), string(ScoreKindProbability)},
	}}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("ambiguousConfidencePools() = %+v, want %+v", got, want)
	}
}
