package framework

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestBatchProfilesWriteIndependentReports(t *testing.T) {
	for _, profile := range []string{"first", "second"} {
		directory := t.TempDir()
		t.Setenv("E2E_REPORT_DIR", directory)
		runner := &Runner{
			opts:     &TestOptions{},
			reporter: NewReportGenerator(profile, "cluster-"+profile),
		}
		runner.finalizeReport(&runState{})
		raw, err := os.ReadFile(filepath.Join(directory, "test-report.json"))
		if err != nil {
			t.Fatal(err)
		}
		var report TestReport
		if err := json.Unmarshal(raw, &report); err != nil {
			t.Fatal(err)
		}
		if report.Profile != profile || report.ClusterName != "cluster-"+profile {
			t.Fatalf("report escaped its profile: %+v", report)
		}
		if _, err := os.Stat(filepath.Join(directory, "test-report.md")); err != nil {
			t.Fatal(err)
		}
		if reportPath("semantic-router-logs.txt") != filepath.Join(directory, "semantic-router-logs.txt") {
			t.Fatal("cluster logs must use the profile's report directory")
		}
	}
	t.Setenv("E2E_REPORT_DIR", "")
	if reportPath("test-report.json") != "test-report.json" {
		t.Fatal("local report path changed")
	}
}
