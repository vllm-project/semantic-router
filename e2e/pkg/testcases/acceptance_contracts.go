package testcases

import (
	"context"
	"errors"
	"fmt"
	"reflect"
	"strings"

	"k8s.io/client-go/kubernetes"
)

const (
	// Conservative floors for the baseline ai-gateway contract.
	// These catch severe behavioral regressions without turning E2E into
	// a model-benchmark gate.
	minBaselineDomainClassificationAccuracy = 60.0
	minBaselineSemanticCacheHitRate         = 40.0
	minBaselinePIIDetectionRate             = 65.0
	minBaselineJailbreakDetectionRate       = 70.0
	minBaselineDecisionPriorityAccuracy     = 75.0
	minBaselineRuleConditionAccuracy        = 80.0
	minBaselineDecisionFallbackAccuracy     = 85.0
	minBaselinePluginChainAccuracy          = 60.0
	minBaselinePluginConfigAccuracy         = 80.0
	minBaselineSequentialStressSuccessRate  = 95.0
	minBaselineProgressiveStressOverallRate = 75.0
)

var progressiveStressStageFloors = []progressiveStressFloor{
	{qps: 10, minimum: 95.0},
	{qps: 20, minimum: 85.0},
	{qps: 50, minimum: 60.0},
}

type acceptanceContract func(details map[string]interface{}) error

type flatRateContract struct {
	actualKey      string
	minimumKey     string
	numeratorKey   string
	denominatorKey string
	minimum        float64
	metricName     string
}

type progressiveStressFloor struct {
	qps     int
	minimum float64
}

const baselineRouterContractProfile = "ai-gateway"

var baselineAcceptanceContracts = map[string]acceptanceContract{
	"domain-classify": applyFlatRateContract(flatRateContract{
		actualKey:      "accuracy_rate",
		minimumKey:     "minimum_accuracy_rate",
		numeratorKey:   "correct_tests",
		denominatorKey: "total_tests",
		minimum:        minBaselineDomainClassificationAccuracy,
		metricName:     "correct classifications",
	}),
	"semantic-cache": applyFlatRateContract(flatRateContract{
		actualKey:      "hit_rate",
		minimumKey:     "minimum_hit_rate",
		numeratorKey:   "cache_hits",
		denominatorKey: "total_requests",
		minimum:        minBaselineSemanticCacheHitRate,
		metricName:     "cache hits",
	}),
	"pii-detection": applyFlatRateContract(flatRateContract{
		actualKey:      "detection_rate",
		minimumKey:     "minimum_detection_rate",
		numeratorKey:   "correct_tests",
		denominatorKey: "total_tests",
		minimum:        minBaselinePIIDetectionRate,
		metricName:     "correct PII detections",
	}),
	"jailbreak-detection": applyFlatRateContract(flatRateContract{
		actualKey:      "detection_rate",
		minimumKey:     "minimum_detection_rate",
		numeratorKey:   "correct_tests",
		denominatorKey: "total_tests",
		minimum:        minBaselineJailbreakDetectionRate,
		metricName:     "correct jailbreak detections",
	}),
	"decision-priority-selection": applyFlatRateContract(flatRateContract{
		actualKey:      "accuracy_rate",
		minimumKey:     "minimum_accuracy_rate",
		numeratorKey:   "correct_tests",
		denominatorKey: "total_tests",
		minimum:        minBaselineDecisionPriorityAccuracy,
		metricName:     "correct priority selections",
	}),
	"rule-condition-logic": applyFlatRateContract(flatRateContract{
		actualKey:      "accuracy_rate",
		minimumKey:     "minimum_accuracy_rate",
		numeratorKey:   "correct_tests",
		denominatorKey: "total_tests",
		minimum:        minBaselineRuleConditionAccuracy,
		metricName:     "correct rule-condition evaluations",
	}),
	"decision-fallback-behavior": applyFlatRateContract(flatRateContract{
		actualKey:      "accuracy_rate",
		minimumKey:     "minimum_accuracy_rate",
		numeratorKey:   "correct_tests",
		denominatorKey: "total_tests",
		minimum:        minBaselineDecisionFallbackAccuracy,
		metricName:     "correct fallback behaviors",
	}),
	"plugin-chain-execution": applyFlatRateContract(flatRateContract{
		actualKey:      "accuracy_rate",
		minimumKey:     "minimum_accuracy_rate",
		numeratorKey:   "correct_tests",
		denominatorKey: "total_tests",
		minimum:        minBaselinePluginChainAccuracy,
		metricName:     "correct plugin-chain behaviors",
	}),
	"plugin-config-variations": applyFlatRateContract(flatRateContract{
		actualKey:      "accuracy_rate",
		minimumKey:     "minimum_accuracy_rate",
		numeratorKey:   "correct_tests",
		denominatorKey: "total_tests",
		minimum:        minBaselinePluginConfigAccuracy,
		metricName:     "correct plugin configurations",
	}),
	"chat-completions-stress-request": applyFlatRateContract(flatRateContract{
		actualKey:      "success_rate",
		minimumKey:     "minimum_success_rate",
		numeratorKey:   "successful",
		denominatorKey: "total_requests",
		minimum:        minBaselineSequentialStressSuccessRate,
		metricName:     "successful requests",
	}),
	"chat-completions-progressive-stress": applyProgressiveStressContract,
}

var acceptanceContractsByProfile = map[string]map[string]acceptanceContract{
	baselineRouterContractProfile: baselineAcceptanceContracts,
}

func wrapWithAcceptanceContract(
	name string,
	fn func(ctx context.Context, client *kubernetes.Clientset, opts TestCaseOptions) error,
) func(ctx context.Context, client *kubernetes.Clientset, opts TestCaseOptions) error {
	if !hasScopedAcceptanceContract(name) {
		return fn
	}

	return func(ctx context.Context, client *kubernetes.Clientset, opts TestCaseOptions) error {
		contract, ok := acceptanceContractForProfile(opts.Profile, name)
		if !ok {
			return fn(ctx, client, opts)
		}

		originalSetDetails := opts.SetDetails
		var details map[string]interface{}

		opts.SetDetails = func(next map[string]interface{}) {
			details = next
		}

		err := fn(ctx, client, opts)
		if contractErr := evaluateAcceptanceContract(name, contract, details); err == nil && contractErr != nil {
			err = contractErr
		}

		if originalSetDetails != nil && details != nil {
			originalSetDetails(details)
		}

		return err
	}
}

func hasScopedAcceptanceContract(name string) bool {
	for _, contracts := range acceptanceContractsByProfile {
		if _, ok := contracts[name]; ok {
			return true
		}
	}

	return false
}

func acceptanceContractForProfile(profile, name string) (acceptanceContract, bool) {
	contracts, ok := acceptanceContractsByProfile[profile]
	if !ok {
		return nil, false
	}

	contract, ok := contracts[name]
	return contract, ok
}

func evaluateAcceptanceContract(name string, contract acceptanceContract, details map[string]interface{}) error {
	if details == nil {
		return fmt.Errorf("%s did not publish structured details required for acceptance evaluation", name)
	}

	if err := contract(details); err != nil {
		return fmt.Errorf("%s failed acceptance contract: %w", name, err)
	}

	return nil
}

func applyFlatRateContract(spec flatRateContract) acceptanceContract {
	return func(details map[string]interface{}) error {
		rate, err := buildAcceptanceRate(details, spec.numeratorKey, spec.denominatorKey)
		if err != nil {
			return err
		}

		rate.addDetails(details, spec.actualKey, spec.minimumKey, spec.minimum)
		return rate.requireMinimum(spec.metricName, spec.minimum)
	}
}

func applyProgressiveStressContract(details map[string]interface{}) error {
	var failures []string
	totalRequests := 0
	totalSuccess := 0

	for _, floor := range progressiveStressStageFloors {
		stageKey := fmt.Sprintf("qps_%d", floor.qps)
		stageDetails, err := nestedDetails(details, stageKey)
		if err != nil {
			failures = append(failures, err.Error())
			continue
		}

		rate, err := buildAcceptanceRate(stageDetails, "successful", "total_requests")
		if err != nil {
			failures = append(failures, fmt.Sprintf("%s: %v", stageKey, err))
			continue
		}

		rate.addDetails(stageDetails, "success_rate", "minimum_success_rate", floor.minimum)
		if err := rate.requireMinimum(
			fmt.Sprintf("%s successful requests", stageKey),
			floor.minimum,
		); err != nil {
			failures = append(failures, err.Error())
		}

		totalRequests += rate.total
		totalSuccess += rate.passed
	}

	overallDetails, err := ensureNestedDetails(details, "overall")
	if err != nil {
		return err
	}

	overallDetails["total_requests"] = totalRequests
	overallDetails["successful"] = totalSuccess
	overallDetails["failed"] = totalRequests - totalSuccess

	overallRate := acceptanceRate{passed: totalSuccess, total: totalRequests}
	overallRate.addDetails(overallDetails, "success_rate", "minimum_success_rate", minBaselineProgressiveStressOverallRate)
	if err := overallRate.requireMinimum("overall successful requests", minBaselineProgressiveStressOverallRate); err != nil {
		failures = append(failures, err.Error())
	}

	if len(failures) > 0 {
		return errors.New(strings.Join(failures, "; "))
	}

	return nil
}

type acceptanceRate struct {
	passed int
	total  int
}

func buildAcceptanceRate(details map[string]interface{}, passedKey, totalKey string) (acceptanceRate, error) {
	passed, err := detailInt(details, passedKey)
	if err != nil {
		return acceptanceRate{}, err
	}

	total, err := detailInt(details, totalKey)
	if err != nil {
		return acceptanceRate{}, err
	}

	return acceptanceRate{
		passed: passed,
		total:  total,
	}, nil
}

func (r acceptanceRate) percent() float64 {
	if r.total == 0 {
		return 0
	}

	return float64(r.passed) / float64(r.total) * 100
}

func (r acceptanceRate) addDetails(details map[string]interface{}, actualKey, minimumKey string, minimum float64) {
	details[actualKey] = formatPercent(r.percent())
	details[minimumKey] = formatPercent(minimum)
}

func (r acceptanceRate) requireMinimum(metricName string, minimum float64) error {
	actual := r.percent()
	if actual < minimum {
		return fmt.Errorf("%d/%d %s (%.2f%%) below minimum %.2f%%", r.passed, r.total, metricName, actual, minimum)
	}

	return nil
}

func nestedDetails(details map[string]interface{}, key string) (map[string]interface{}, error) {
	value, ok := details[key]
	if !ok {
		return nil, fmt.Errorf("missing details[%q]", key)
	}

	nested, ok := value.(map[string]interface{})
	if !ok {
		return nil, fmt.Errorf("details[%q] is %T, want map[string]interface{}", key, value)
	}

	return nested, nil
}

func ensureNestedDetails(details map[string]interface{}, key string) (map[string]interface{}, error) {
	if value, ok := details[key]; ok {
		nested, ok := value.(map[string]interface{})
		if !ok {
			return nil, fmt.Errorf("details[%q] is %T, want map[string]interface{}", key, value)
		}

		return nested, nil
	}

	nested := make(map[string]interface{})
	details[key] = nested
	return nested, nil
}

func detailInt(details map[string]interface{}, key string) (int, error) {
	value, ok := details[key]
	if !ok {
		return 0, fmt.Errorf("missing details[%q]", key)
	}

	number := reflect.ValueOf(value)
	switch number.Kind() {
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		return int(number.Int()), nil
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		return int(number.Uint()), nil
	case reflect.Float32, reflect.Float64:
		return int(number.Float()), nil
	}

	return 0, fmt.Errorf("details[%q] is %T, want integer count", key, value)
}

func formatPercent(value float64) string {
	return fmt.Sprintf("%.2f%%", value)
}
