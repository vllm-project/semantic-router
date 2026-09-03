package compatibility

import (
	"os"
	"strings"
	"testing"
)

func TestCandleClassifierSubjectDigest(t *testing.T) {
	subject := testSubject()
	first, err := subject.Digest()
	if err != nil {
		t.Fatalf("Digest() error = %v", err)
	}
	second, err := subject.Digest()
	if err != nil {
		t.Fatalf("Digest() repeated error = %v", err)
	}
	if first != second {
		t.Fatalf("Digest() repeated = %q, want %q", second, first)
	}

	mutations := map[string]func(*CandleClassifierSubject){
		"artifact revision": func(value *CandleClassifierSubject) { value.ArtifactRevision = "revision-2" },
		"artifact digest":   func(value *CandleClassifierSubject) { value.ArtifactDigest = "sha256:" + strings.Repeat("b", 64) },
		"task contract":     func(value *CandleClassifierSubject) { value.TaskContract = "label_distribution.v2" },
		"connector":         func(value *CandleClassifierSubject) { value.Connector = "sr.candle.embedded.v2" },
		"precision":         func(value *CandleClassifierSubject) { value.Precision = "f16" },
		"provider":          func(value *CandleClassifierSubject) { value.Provider = "candle-cuda" },
		"device profile":    func(value *CandleClassifierSubject) { value.DeviceProfile = "linux/amd64/cuda" },
		"router revision":   func(value *CandleClassifierSubject) { value.RouterRevision = "router-revision-2" },
		"label order":       func(value *CandleClassifierSubject) { value.Labels = []string{"JAILBREAK", "SAFE"} },
	}
	for name, mutate := range mutations {
		t.Run(name, func(t *testing.T) {
			changed := subject
			changed.Labels = append([]string(nil), subject.Labels...)
			mutate(&changed)
			digest, err := changed.Digest()
			if err != nil {
				t.Fatalf("Digest() error = %v", err)
			}
			if digest == first {
				t.Fatalf("Digest() did not change from %q", first)
			}
		})
	}
}

func TestCandleClassifierSubjectValidation(t *testing.T) {
	tests := []struct {
		name    string
		mutate  func(*CandleClassifierSubject)
		wantErr string
	}{
		{"missing revision", func(value *CandleClassifierSubject) { value.ArtifactRevision = "" }, "artifact_revision is required"},
		{"invalid digest", func(value *CandleClassifierSubject) { value.ArtifactDigest = "sha256:not-a-digest" }, "artifact_digest must be a lowercase sha256 digest"},
		{"too few labels", func(value *CandleClassifierSubject) { value.Labels = []string{"SAFE"} }, "at least two"},
		{"duplicate labels", func(value *CandleClassifierSubject) { value.Labels = []string{"SAFE", "SAFE"} }, "duplicated"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			subject := testSubject()
			test.mutate(&subject)
			_, err := subject.Digest()
			if err == nil || !strings.Contains(err.Error(), test.wantErr) {
				t.Fatalf("Digest() error = %v, want containing %q", err, test.wantErr)
			}
		})
	}
}

func TestReceiptValidation(t *testing.T) {
	valid, err := NewReceipt(testSubject(), testSuiteDigest(), testChecks())
	if err != nil {
		t.Fatalf("NewReceipt() error = %v", err)
	}

	tests := []struct {
		name    string
		mutate  func(*Receipt)
		wantErr string
	}{
		{"subject mismatch", func(receipt *Receipt) { receipt.SubjectDigest = "sha256:" + strings.Repeat("0", 64) }, "does not match"},
		{"invalid suite digest", func(receipt *Receipt) { receipt.QualificationSuiteDigest = "synthetic" }, "qualification_suite_digest"},
		{"empty check name", func(receipt *Receipt) { receipt.Checks = []CheckOutcome{{Passed: true}} }, "name is required"},
		{"duplicate check name", func(receipt *Receipt) {
			receipt.Checks = []CheckOutcome{{Name: "label_parity"}, {Name: "label_parity"}}
		}, "duplicated"},
		{"missing required check", func(receipt *Receipt) {
			receipt.Checks = receipt.Checks[:len(receipt.Checks)-1]
		}, "unavailable_behavior\" is required"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			receipt := valid
			receipt.Checks = append([]CheckOutcome(nil), valid.Checks...)
			test.mutate(&receipt)
			if err := receipt.Validate(); err == nil || !strings.Contains(err.Error(), test.wantErr) {
				t.Fatalf("Validate() error = %v, want containing %q", err, test.wantErr)
			}
		})
	}
}

func TestReceiptPreservesFailedCheck(t *testing.T) {
	checks := testChecks()
	checks[0].Passed = false
	checks[0].Details = "observed mismatch"
	receipt, err := NewReceipt(testSubject(), testSuiteDigest(), checks)
	if err != nil {
		t.Fatalf("NewReceipt() error = %v", err)
	}
	if receipt.Checks[0].Passed || receipt.Checks[0].Details != "observed mismatch" {
		t.Fatalf("failed check = %+v, want preserved failure", receipt.Checks[0])
	}
}

func TestParseReceiptRejectsUnknownFields(t *testing.T) {
	data, err := os.ReadFile("testdata/local-candle-cpu-v1.json")
	if err != nil {
		t.Fatalf("ReadFile() error = %v", err)
	}
	data = []byte(strings.Replace(string(data), `"schema_version":`, `"unknown":true,"schema_version":`, 1))
	if _, err := ParseReceipt(data); err == nil || !strings.Contains(err.Error(), "unknown field") {
		t.Fatalf("ParseReceipt() error = %v, want unknown field", err)
	}
}

func TestLocalCandleCPUReceiptFixture(t *testing.T) {
	data, err := os.ReadFile("testdata/local-candle-cpu-v1.json")
	if err != nil {
		t.Fatalf("ReadFile() error = %v", err)
	}
	receipt, err := ParseReceipt(data)
	if err != nil {
		t.Fatalf("ParseReceipt() error = %v", err)
	}
	if receipt.Subject.Connector != CandleConnectorV1 {
		t.Fatalf("connector = %q, want %q", receipt.Subject.Connector, CandleConnectorV1)
	}
	if receipt.Subject.TaskContract != LabelDistributionContractV1 {
		t.Fatalf("task contract = %q, want %q", receipt.Subject.TaskContract, LabelDistributionContractV1)
	}
	if receipt.Subject.DeviceProfile != "linux/amd64/cpu" {
		t.Fatalf("device profile = %q, want linux/amd64/cpu", receipt.Subject.DeviceProfile)
	}
	if len(receipt.Subject.Labels) != 2 || receipt.Subject.Labels[0] != "SAFE" || receipt.Subject.Labels[1] != "JAILBREAK" {
		t.Fatalf("labels = %v, want [SAFE JAILBREAK]", receipt.Subject.Labels)
	}
	if len(receipt.Checks) != len(requiredCandleChecks) {
		t.Fatalf("checks = %d, want %d", len(receipt.Checks), len(requiredCandleChecks))
	}
	for _, check := range receipt.Checks {
		if !strings.Contains(check.Details, "not conformance evidence") {
			t.Fatalf("check %q details = %q, want synthetic-evidence warning", check.Name, check.Details)
		}
	}
}

func testSubject() CandleClassifierSubject {
	return CandleClassifierSubject{
		SchemaVersion:    SubjectSchemaVersionV1,
		ArtifactRevision: "synthetic-model-revision-001",
		ArtifactDigest:   "sha256:" + strings.Repeat("a", 64),
		TaskContract:     LabelDistributionContractV1,
		Connector:        CandleConnectorV1,
		Precision:        "float32",
		Provider:         "cpu",
		DeviceProfile:    "linux/amd64/cpu",
		RouterRevision:   "synthetic-router-revision-001",
		Labels:           []string{"SAFE", "JAILBREAK"},
	}
}

func testChecks() []CheckOutcome {
	checks := make([]CheckOutcome, 0, len(requiredCandleChecks))
	for _, name := range requiredCandleChecks {
		checks = append(checks, CheckOutcome{Name: name, Passed: true})
	}
	return checks
}

func testSuiteDigest() string {
	return "sha256:" + strings.Repeat("b", 64)
}
