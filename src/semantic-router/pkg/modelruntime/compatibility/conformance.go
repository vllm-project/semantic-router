package compatibility

import (
	"fmt"
	"math"
	"strings"
	"time"
)

const probabilityTolerance = 0.001

// ClassificationResult is the observable label-distribution contract at the
// Candle FFI seam.
type ClassificationResult struct {
	Class         int
	Confidence    float32
	Probabilities []float32
	NumClasses    int
}

// CandleRuntime is implemented by the active native binding and by model-free tests.
type CandleRuntime interface {
	Initialize(modelPath string, numClasses int) error
	Classify(input string) (ClassificationResult, error)
}

// QualifyLocalCandleCPU executes one offline conformance run. Failed checks are
// evidence and therefore produce a valid receipt; invalid inputs return an error.
func QualifyLocalCandleCPU(
	subject Subject,
	modelPath string,
	suite QualificationSuite,
	runtime CandleRuntime,
) (Receipt, error) {
	if runtime == nil {
		return Receipt{}, fmt.Errorf("candle qualification runtime is required")
	}
	if err := validateLocalCandleCPUSubject(subject); err != nil {
		return Receipt{}, err
	}
	suiteDigest, err := suite.Digest(subject.Labels)
	if err != nil {
		return Receipt{}, err
	}
	artifactDigest, err := DigestLocalCandleArtifact(modelPath)
	if err != nil {
		return Receipt{}, err
	}
	if subject.ArtifactDigest != artifactDigest {
		return Receipt{}, fmt.Errorf(
			"compatibility subject artifact_digest %q does not match local Candle artifact %q",
			subject.ArtifactDigest,
			artifactDigest,
		)
	}

	unavailable := checkUnavailable(runtime, suite.DeadlineBehavior.Input)
	if err := runtime.Initialize(modelPath, len(subject.Labels)); err != nil {
		detail := fmt.Sprintf("Candle initialization failed: %v", err)
		return NewReceipt(subject, suiteDigest, []CheckOutcome{
			{Name: CheckLabelParity, Passed: false, Details: detail},
			{Name: CheckInputBounds, Passed: false, Details: detail},
			{Name: CheckDeadlineBehavior, Passed: false, Details: detail},
			unavailable,
		})
	}

	return NewReceipt(subject, suiteDigest, []CheckOutcome{
		checkLabelParity(runtime, subject.Labels, suite.LabelParity),
		checkInputBounds(runtime, subject.Labels, suite.InputBounds),
		checkDeadline(runtime, subject.Labels, suite.DeadlineBehavior),
		unavailable,
	})
}

func validateLocalCandleCPUSubject(subject Subject) error {
	if err := subject.Validate(); err != nil {
		return err
	}
	required := []struct {
		name string
		got  string
		want string
	}{
		{"task_contract", subject.TaskContract, LabelDistributionContractV1},
		{"connector", subject.Connector, CandleConnectorV1},
		{"precision", subject.Precision, "float32"},
		{"provider", subject.Provider, "cpu"},
	}
	for _, value := range required {
		if value.got != value.want {
			return fmt.Errorf(
				"local Candle CPU subject %s must be %q, got %q",
				value.name,
				value.want,
				value.got,
			)
		}
	}
	if !strings.HasSuffix(subject.DeviceProfile, "/cpu") {
		return fmt.Errorf(
			"local Candle CPU subject device_profile must end with %q, got %q",
			"/cpu",
			subject.DeviceProfile,
		)
	}
	return nil
}

// FailedCheckNames returns failed outcome names in receipt order.
func FailedCheckNames(receipt Receipt) []string {
	failed := make([]string, 0, len(receipt.Checks))
	for _, check := range receipt.Checks {
		if !check.Passed {
			failed = append(failed, check.Name)
		}
	}
	return failed
}

func checkUnavailable(runtime CandleRuntime, input string) CheckOutcome {
	if _, err := runtime.Classify(input); err != nil {
		return CheckOutcome{
			Name:    CheckUnavailableBehavior,
			Passed:  true,
			Details: "pre-initialization inference failed closed",
		}
	}
	return CheckOutcome{
		Name:    CheckUnavailableBehavior,
		Passed:  false,
		Details: "pre-initialization inference returned a classification",
	}
}

func checkLabelParity(
	runtime CandleRuntime,
	labels []string,
	probes []LabelProbe,
) CheckOutcome {
	for _, probe := range probes {
		result, err := runtime.Classify(probe.Input)
		if err != nil {
			return failedProbe(CheckLabelParity, probe.Name, err)
		}
		if err := validateDistribution(result, len(labels)); err != nil {
			return failedProbe(CheckLabelParity, probe.Name, err)
		}
		if got := labels[result.Class]; got != probe.WantLabel {
			return failedProbe(
				CheckLabelParity,
				probe.Name,
				fmt.Errorf("predicted label %q, want %q", got, probe.WantLabel),
			)
		}
	}
	return CheckOutcome{
		Name:    CheckLabelParity,
		Passed:  true,
		Details: fmt.Sprintf("%d label probes matched ordered labels and normalized distributions", len(probes)),
	}
}

func checkInputBounds(
	runtime CandleRuntime,
	labels []string,
	probes []InputBoundaryProbe,
) CheckOutcome {
	for _, probe := range probes {
		result, err := runtime.Classify(probe.Input)
		switch {
		case probe.WantError && err == nil:
			return failedProbe(CheckInputBounds, probe.Name, fmt.Errorf("classification succeeded, want error"))
		case probe.WantError:
			continue
		case err != nil:
			return failedProbe(CheckInputBounds, probe.Name, err)
		}
		if err := validateDistribution(result, len(labels)); err != nil {
			return failedProbe(CheckInputBounds, probe.Name, err)
		}
	}
	return CheckOutcome{
		Name:    CheckInputBounds,
		Passed:  true,
		Details: fmt.Sprintf("%d input-bound probes matched explicit expectations", len(probes)),
	}
}

func checkDeadline(
	runtime CandleRuntime,
	labels []string,
	probe DeadlineProbe,
) CheckOutcome {
	started := time.Now()
	result, err := runtime.Classify(probe.Input)
	elapsed := time.Since(started)
	if err != nil {
		return CheckOutcome{Name: CheckDeadlineBehavior, Passed: false, Details: err.Error()}
	}
	if err := validateDistribution(result, len(labels)); err != nil {
		return CheckOutcome{Name: CheckDeadlineBehavior, Passed: false, Details: err.Error()}
	}
	budget := time.Duration(probe.MaxDurationMillis) * time.Millisecond
	if elapsed > budget {
		return CheckOutcome{
			Name:   CheckDeadlineBehavior,
			Passed: false,
			Details: fmt.Sprintf(
				"synchronous inference exceeded the configured %dms completion budget",
				probe.MaxDurationMillis,
			),
		}
	}
	return CheckOutcome{
		Name:   CheckDeadlineBehavior,
		Passed: true,
		Details: fmt.Sprintf(
			"synchronous inference completed within the configured %dms budget; cancellation is not claimed",
			probe.MaxDurationMillis,
		),
	}
}

func validateDistribution(result ClassificationResult, labelCount int) error {
	if err := validateDistributionShape(result, labelCount); err != nil {
		return err
	}
	confidence := float64(result.Confidence)
	if err := validateProbability("confidence", confidence); err != nil {
		return err
	}
	if err := validateProbabilityValues(result.Probabilities); err != nil {
		return err
	}
	return validateTopPrediction(result, confidence)
}

func validateDistributionShape(result ClassificationResult, labelCount int) error {
	if result.Class < 0 || result.Class >= labelCount {
		return fmt.Errorf("class index %d is outside %d ordered labels", result.Class, labelCount)
	}
	if result.NumClasses != labelCount {
		return fmt.Errorf("reported %d classes, want %d", result.NumClasses, labelCount)
	}
	if len(result.Probabilities) != labelCount {
		return fmt.Errorf("returned %d probabilities, want %d", len(result.Probabilities), labelCount)
	}
	return nil
}

func validateProbabilityValues(probabilities []float32) error {
	sum := 0.0
	for index, probability := range probabilities {
		value := float64(probability)
		if err := validateProbability(fmt.Sprintf("probability[%d]", index), value); err != nil {
			return err
		}
		sum += value
	}
	if math.Abs(sum-1) > probabilityTolerance {
		return fmt.Errorf("probabilities sum to %.6f, want 1", sum)
	}
	return nil
}

func validateProbability(name string, value float64) error {
	if math.IsNaN(value) || math.IsInf(value, 0) || value < 0 || value > 1 {
		return fmt.Errorf("%s %v is not finite and within [0,1]", name, value)
	}
	return nil
}

func validateTopPrediction(result ClassificationResult, confidence float64) error {
	selected := float64(result.Probabilities[result.Class])
	if math.Abs(selected-confidence) > probabilityTolerance {
		return fmt.Errorf(
			"confidence %.6f does not match class probability %.6f",
			confidence,
			selected,
		)
	}
	for index, probability := range result.Probabilities {
		if float64(probability) > selected+probabilityTolerance {
			return fmt.Errorf(
				"class %d probability %.6f is below class %d probability %.6f",
				result.Class,
				selected,
				index,
				probability,
			)
		}
	}
	return nil
}

func failedProbe(check string, probe string, err error) CheckOutcome {
	return CheckOutcome{
		Name:    check,
		Passed:  false,
		Details: fmt.Sprintf("probe %q: %v", probe, err),
	}
}
