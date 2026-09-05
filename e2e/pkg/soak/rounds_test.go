package soak

import (
	"strings"
	"testing"
)

func TestFinalizeRecordsLowSuccessRate(t *testing.T) {
	newRunner := func(t *testing.T) *Runner {
		t.Helper()
		r, err := NewRunner(NewPlan(Config{OutDir: t.TempDir(), MetricsURL: "http://127.0.0.1:0/metrics", PprofURL: "http://127.0.0.1:0"}))
		if err != nil {
			t.Fatal(err)
		}
		return r
	}
	hasBrokenStackNote := func(notes []string) bool {
		for _, n := range notes {
			if strings.Contains(n, "broken stack") {
				return true
			}
		}
		return false
	}

	r := newRunner(t)
	r.summary.Rounds = []RoundStats{{TotalRequests: 100, Successful: 0, Failed: 100}}
	if err := r.finalize(); err != nil {
		t.Fatal(err)
	}
	if r.summary.Measured.Total != 100 {
		t.Errorf("Measured.Total = %d, want 100", r.summary.Measured.Total)
	}
	if r.summary.Measured.SuccessRate != 0 {
		t.Errorf("Measured.SuccessRate = %v, want 0", r.summary.Measured.SuccessRate)
	}
	if !hasBrokenStackNote(r.summary.Notes) {
		t.Errorf("Notes = %v, want an entry mentioning a broken stack", r.summary.Notes)
	}

	r = newRunner(t)
	r.summary.Rounds = []RoundStats{{TotalRequests: 100, Successful: 100}}
	if err := r.finalize(); err != nil {
		t.Fatal(err)
	}
	if r.summary.Measured.SuccessRate != 1 {
		t.Errorf("Measured.SuccessRate = %v, want 1", r.summary.Measured.SuccessRate)
	}
	if hasBrokenStackNote(r.summary.Notes) {
		t.Errorf("Notes = %v, want no broken-stack entry on a fully-served run", r.summary.Notes)
	}
}

// A zero or negative -high-card-ids must be clamped, not divided by, so the
// high-cardinality round cannot panic after hours of completed soak rounds.
func TestNewClientClampsHighCardIDs(t *testing.T) {
	for _, in := range []int{0, -5} {
		if got := NewClient("http://127.0.0.1:8801", "MoM", 10, in, false).highCardIDs; got != 1 {
			t.Errorf("NewClient(highCardIDs=%d).highCardIDs = %d, want 1", in, got)
		}
	}
}

func TestResponseModeName(t *testing.T) {
	if got := responseModeName(false); got != "buffered" {
		t.Errorf("responseModeName(false) = %q, want buffered", got)
	}
	if got := responseModeName(true); got != "streaming" {
		t.Errorf("responseModeName(true) = %q, want streaming", got)
	}
}
