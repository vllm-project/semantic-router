package sessionbudget

import "testing"

func TestEvaluate(t *testing.T) {
	def := DefaultThresholds() // {1.0, 1.5, 2.0, 3.0}
	cases := []struct {
		name       string
		cumulative int64
		budget     int64
		want       Stage
	}{
		{"disabled budget zero", 10_000, 0, StageNone},
		{"negative budget disabled", 10_000, -1, StageNone},
		{"zero cumulative", 0, 1000, StageNone},
		{"under budget", 500, 1000, StageNone},
		{"at shape boundary", 1000, 1000, StageShapeTools},
		{"shape band", 1400, 1000, StageShapeTools},
		{"compress band", 1600, 1000, StageCompress},
		{"downgrade band", 2200, 1000, StageDowngrade},
		{"terminate band", 3000, 1000, StageTerminate},
		{"well over", 99_999, 1000, StageTerminate},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, ratio := Evaluate(tc.cumulative, tc.budget, def)
			if got != tc.want {
				t.Fatalf("Evaluate(%d, %d) stage = %v, want %v", tc.cumulative, tc.budget, got, tc.want)
			}
			if tc.budget > 0 && tc.cumulative > 0 {
				want := float64(tc.cumulative) / float64(tc.budget)
				if ratio != want {
					t.Fatalf("ratio = %v, want %v", ratio, want)
				}
			}
		})
	}
}

func TestStageString(t *testing.T) {
	cases := map[Stage]string{
		StageNone:       "none",
		StageShapeTools: "shape_tools",
		StageCompress:   "compress",
		StageDowngrade:  "downgrade",
		StageTerminate:  "terminate",
	}
	for stage, want := range cases {
		if got := stage.String(); got != want {
			t.Fatalf("Stage(%d).String() = %q, want %q", stage, got, want)
		}
	}
}

func TestResolveThresholdsZeroFallsBack(t *testing.T) {
	got := ResolveThresholds(Thresholds{Compress: 1.8}) // only compress set
	def := DefaultThresholds()
	if got.ShapeTools != def.ShapeTools || got.Downgrade != def.Downgrade || got.Terminate != def.Terminate {
		t.Fatalf("zero fields should fall back to defaults, got %+v", got)
	}
	if got.Compress != 1.8 {
		t.Fatalf("explicit field should win, got Compress=%v", got.Compress)
	}
}
