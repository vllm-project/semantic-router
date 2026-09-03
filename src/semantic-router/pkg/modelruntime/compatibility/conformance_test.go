package compatibility

import (
	"errors"
	"math"
	"os"
	"strings"
	"testing"
	"time"
)

type fakeCandleRuntime struct {
	initialized bool
	initErr     error
	results     map[string]ClassificationResult
	errors      map[string]error
	delay       time.Duration
}

func (f *fakeCandleRuntime) Initialize(_ string, _ int, useCPU bool) error {
	if !useCPU {
		return errors.New("expected CPU initialization")
	}
	if f.initErr != nil {
		return f.initErr
	}
	f.initialized = true
	return nil
}

func (f *fakeCandleRuntime) Classify(input string) (ClassificationResult, error) {
	if !f.initialized {
		return ClassificationResult{}, errors.New("classifier unavailable")
	}
	if f.delay > 0 {
		time.Sleep(f.delay)
	}
	if err := f.errors[input]; err != nil {
		return ClassificationResult{}, err
	}
	result, exists := f.results[input]
	if !exists {
		return ClassificationResult{}, errors.New("missing fake result")
	}
	return result, nil
}

func TestQualifyLocalCandleCPUPassesObservedContract(t *testing.T) {
	artifact := writeTestCandleArtifact(t, map[string]string{
		"config.json":       "config",
		"tokenizer.json":    "tokenizer",
		"model.safetensors": "weights",
	})
	subject := qualificationSubject(t, artifact)
	runtime := passingFakeRuntime()

	receipt, err := QualifyLocalCandleCPU(subject, artifact, qualificationSuite(), runtime)
	if err != nil {
		t.Fatalf("QualifyLocalCandleCPU() error = %v", err)
	}
	if failed := FailedCheckNames(receipt); len(failed) != 0 {
		t.Fatalf("FailedCheckNames() = %v, want none", failed)
	}
	if !runtime.initialized {
		t.Fatal("runtime was not initialized")
	}
}

func TestQualifyLocalCandleCPURecordsMalformedDistribution(t *testing.T) {
	artifact := writeTestCandleArtifact(t, map[string]string{
		"config.json":       "config",
		"tokenizer.json":    "tokenizer",
		"model.safetensors": "weights",
	})
	runtime := passingFakeRuntime()
	runtime.results["hello"] = ClassificationResult{
		Class:         1,
		Confidence:    0.8,
		Probabilities: nil,
		NumClasses:    0,
	}

	receipt, err := QualifyLocalCandleCPU(
		qualificationSubject(t, artifact),
		artifact,
		qualificationSuite(),
		runtime,
	)
	if err != nil {
		t.Fatalf("QualifyLocalCandleCPU() error = %v", err)
	}
	failed := FailedCheckNames(receipt)
	if len(failed) != 1 || failed[0] != CheckLabelParity {
		t.Fatalf("FailedCheckNames() = %v, want [%s]", failed, CheckLabelParity)
	}
	if !strings.Contains(receipt.Checks[0].Details, "reported 0 classes") {
		t.Fatalf("label parity details = %q", receipt.Checks[0].Details)
	}
}

func TestQualifyLocalCandleCPURecordsInitializationFailure(t *testing.T) {
	artifact := writeTestCandleArtifact(t, map[string]string{
		"config.json":       "config",
		"tokenizer.json":    "tokenizer",
		"model.safetensors": "weights",
	})
	runtime := passingFakeRuntime()
	runtime.initErr = errors.New("load failed")

	receipt, err := QualifyLocalCandleCPU(
		qualificationSubject(t, artifact),
		artifact,
		qualificationSuite(),
		runtime,
	)
	if err != nil {
		t.Fatalf("QualifyLocalCandleCPU() error = %v", err)
	}
	want := []string{CheckLabelParity, CheckInputBounds, CheckDeadlineBehavior}
	if failed := FailedCheckNames(receipt); !slicesEqual(failed, want) {
		t.Fatalf("FailedCheckNames() = %v, want %v", failed, want)
	}
	if !receipt.Checks[3].Passed {
		t.Fatalf("unavailable outcome = %+v, want pass", receipt.Checks[3])
	}
}

func TestQualifyLocalCandleCPURejectsArtifactMismatch(t *testing.T) {
	artifact := writeTestCandleArtifact(t, map[string]string{
		"config.json":       "config",
		"tokenizer.json":    "tokenizer",
		"model.safetensors": "weights",
	})
	subject := qualificationSubject(t, artifact)
	subject.ArtifactDigest = "sha256:" + strings.Repeat("0", 64)

	_, err := QualifyLocalCandleCPU(subject, artifact, qualificationSuite(), passingFakeRuntime())
	if err == nil || !strings.Contains(err.Error(), "does not match local Candle artifact") {
		t.Fatalf("QualifyLocalCandleCPU() error = %v, want artifact mismatch", err)
	}
}

func TestQualificationSuiteValidation(t *testing.T) {
	tests := []struct {
		name    string
		mutate  func(*QualificationSuite)
		wantErr string
	}{
		{"unknown label", func(suite *QualificationSuite) { suite.LabelParity[0].WantLabel = "OTHER" }, "not in subject labels"},
		{"duplicate bound", func(suite *QualificationSuite) { suite.InputBounds[1].Name = suite.InputBounds[0].Name }, "duplicated"},
		{"missing deadline", func(suite *QualificationSuite) { suite.DeadlineBehavior.MaxDurationMillis = 0 }, "must be positive"},
		{"overflowing deadline", func(suite *QualificationSuite) { suite.DeadlineBehavior.MaxDurationMillis = math.MaxInt64 }, "exceeds time.Duration"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			suite := qualificationSuite()
			test.mutate(&suite)
			if err := suite.Validate([]string{"LABEL_0", "LABEL_1"}); err == nil || !strings.Contains(err.Error(), test.wantErr) {
				t.Fatalf("Validate() error = %v, want containing %q", err, test.wantErr)
			}
		})
	}
}

func TestPinnedTinyBERTCPUSuite(t *testing.T) {
	data, err := os.ReadFile("testdata/tiny-random-bert-cpu-suite-v1.json")
	if err != nil {
		t.Fatalf("ReadFile() error = %v", err)
	}
	suite, err := ParseQualificationSuite(data, []string{"LABEL_0", "LABEL_1"})
	if err != nil {
		t.Fatalf("ParseQualificationSuite() error = %v", err)
	}
	if len(suite.InputBounds) != 2 || len(suite.InputBounds[1].Input) < 512 {
		t.Fatalf("input-bound probes do not include an oversized vector: %+v", suite.InputBounds)
	}
	first, err := suite.Digest([]string{"LABEL_0", "LABEL_1"})
	if err != nil {
		t.Fatalf("Digest() error = %v", err)
	}
	const want = "sha256:ff187849f56c72f4b5347836a329e0d17dfc08119d17cea972c53b0ebe2de128"
	if first != want {
		t.Fatalf("Digest() = %q, want golden %q", first, want)
	}
	second, err := suite.Digest([]string{"LABEL_0", "LABEL_1"})
	if err != nil {
		t.Fatalf("Digest() repeated error = %v", err)
	}
	if first != second {
		t.Fatalf("Digest() repeated = %q, want %q", second, first)
	}
	suite.LabelParity[0].Input += " changed"
	changed, err := suite.Digest([]string{"LABEL_0", "LABEL_1"})
	if err != nil {
		t.Fatalf("Digest() changed error = %v", err)
	}
	if changed == first {
		t.Fatalf("Digest() did not change from %q", first)
	}
}

func TestDeadlineCheckRecordsBudgetFailure(t *testing.T) {
	runtime := passingFakeRuntime()
	runtime.initialized = true
	runtime.delay = 5 * time.Millisecond
	probe := DeadlineProbe{Input: "deadline", MaxDurationMillis: 1}
	outcome := checkDeadline(runtime, []string{"LABEL_0", "LABEL_1"}, probe)
	if outcome.Passed || !strings.Contains(outcome.Details, "exceeded") {
		t.Fatalf("checkDeadline() = %+v, want exceeded failure", outcome)
	}
}

func passingFakeRuntime() *fakeCandleRuntime {
	classZero := ClassificationResult{
		Class:         0,
		Confidence:    0.75,
		Probabilities: []float32{0.75, 0.25},
		NumClasses:    2,
	}
	classOne := ClassificationResult{
		Class:         1,
		Confidence:    0.8,
		Probabilities: []float32{0.2, 0.8},
		NumClasses:    2,
	}
	return &fakeCandleRuntime{
		results: map[string]ClassificationResult{
			"hello":    classOne,
			"":         classZero,
			"deadline": classOne,
		},
		errors: map[string]error{"oversized": errors.New("input exceeds bound")},
	}
}

func qualificationSubject(t *testing.T, artifact string) CandleClassifierSubject {
	t.Helper()
	digest, err := DigestLocalCandleArtifact(artifact)
	if err != nil {
		t.Fatalf("DigestLocalCandleArtifact() error = %v", err)
	}
	return CandleClassifierSubject{
		SchemaVersion:    SubjectSchemaVersionV1,
		ArtifactRevision: "model-commit",
		ArtifactDigest:   digest,
		TaskContract:     LabelDistributionContractV1,
		Connector:        CandleConnectorV1,
		Precision:        "float32",
		Provider:         "cpu",
		DeviceProfile:    "linux/amd64/cpu",
		RouterRevision:   "router-commit",
		Labels:           []string{"LABEL_0", "LABEL_1"},
	}
}

func qualificationSuite() QualificationSuite {
	return QualificationSuite{
		SchemaVersion: QualificationSuiteSchemaVersionV1,
		LabelParity: []LabelProbe{
			{Name: "class-one", Input: "hello", WantLabel: "LABEL_1"},
		},
		InputBounds: []InputBoundaryProbe{
			{Name: "empty", Input: "", WantError: false},
			{Name: "oversized", Input: "oversized", WantError: true},
		},
		DeadlineBehavior: DeadlineProbe{Input: "deadline", MaxDurationMillis: 1000},
	}
}

func slicesEqual(left []string, right []string) bool {
	if len(left) != len(right) {
		return false
	}
	for index := range left {
		if left[index] != right[index] {
			return false
		}
	}
	return true
}
