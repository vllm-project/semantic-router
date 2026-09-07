package evaluationplane

import (
	"fmt"
	"math"
	"regexp"
	"sort"
)

const maxModelPoolDiversityArms = 64

var modelPoolDynamicMetricIDPattern = regexp.MustCompile(`^model_pool\.arm\.([A-Za-z0-9][A-Za-z0-9_-]{0,172})\.(quality|success_rate|marginal_contribution)$`)

var modelPoolStaticMetricIDs = []string{
	"model_pool.all_arm_failure_rate", "model_pool.arm_count", "model_pool.best_single_quality",
	"model_pool.mean_pairwise_failure_jaccard", "model_pool.oracle_gain", "model_pool.oracle_quality",
	"model_pool.pareto_dominated_arm_count", "model_pool.pareto_evaluable_arm_count",
	"model_pool.quality_cost_shared_support_cases", "model_pool.quality_cost_shared_support_fraction",
	"model_pool.quality_dominated_arm_count", "model_pool.quality_shared_support_cases",
	"model_pool.quality_shared_support_fraction", "model_pool.selection_arm_coverage",
	"model_pool.selection_entropy_bits", "model_pool.unique_win_rate", "model_pool.unique_wins",
	"model_pool.worst_arm_reliability",
}

func isCanonicalModelPoolMetricID(id string) bool {
	for _, staticID := range modelPoolStaticMetricIDs {
		if id == staticID {
			return true
		}
	}
	_, _, ok := parseCanonicalModelPoolArmMetricID(id)
	return ok
}

func modelPoolMetricArmSegment(armID string) string {
	segment, err := EncodeMetricAnalysisSubjectID(armID)
	if err != nil {
		return ""
	}
	return segment
}

func parseCanonicalModelPoolArmMetricID(id string) (armID, measure string, ok bool) {
	parts := modelPoolDynamicMetricIDPattern.FindStringSubmatch(id)
	if len(parts) != 3 {
		return "", "", false
	}
	decoded, err := DecodeMetricAnalysisSubjectID(parts[1])
	if err != nil || !evidenceIDPattern.MatchString(decoded) || modelPoolMetricArmSegment(decoded) != parts[1] {
		return "", "", false
	}
	return decoded, parts[2], true
}

func modelPoolArmMetricID(armID, measure string) string {
	return "model_pool.arm." + modelPoolMetricArmSegment(armID) + "." + measure
}

type modelPoolMissingReason string

const (
	modelPoolNonAuthoritative   modelPoolMissingReason = "non_authoritative"
	modelPoolMissingArmCell     modelPoolMissingReason = "missing_arm_cell"
	modelPoolUngradedSuccess    modelPoolMissingReason = "ungraded_success"
	modelPoolUnavailableRecord  modelPoolMissingReason = "unavailable_record"
	modelPoolMissingRuntimeCost modelPoolMissingReason = "missing_runtime_cost"
	modelPoolMissingSelection   modelPoolMissingReason = "missing_joint_selection"
)

// modelPoolReductionInput contains only immutable, already validated evidence.
// The caller supplies the frozen matrix instead of inferring it from worker
// rows, so a partial report cannot silently redefine the analysis cohort.
type modelPoolReductionInput struct {
	FrozenArmIDs   []string
	PlannedCaseIDs []string
	Authoritative  bool
	PoolRecords    []executionRecordEvidence
	JointRecords   []executionRecordEvidence
}

// modelPoolMetricEvidence is the internal, server-reduced form that a later
// sealing integration can compare with public Metric values and provenance.
type modelPoolMetricEvidence struct {
	ID                  string
	Value               *float64
	SampleCount         int
	MissingReasonCounts map[modelPoolMissingReason]int
}

type modelPoolCell struct {
	present      bool
	successKnown bool
	success      bool
	quality      *float64
	runtimeCost  *float64
}

type modelPoolSupport struct {
	qualityReasons           map[modelPoolMissingReason]int
	successReasons           map[modelPoolMissingReason]int
	costReasons              map[modelPoolMissingReason]int
	qualityCompleteCases     []string
	qualityCostCompleteCases []string
	successComplete          bool
	allArmFailures           int
}

func reduceAuthoritativeModelPoolMetrics(input modelPoolReductionInput) ([]modelPoolMetricEvidence, error) {
	arms, err := canonicalModelPoolIDs(input.FrozenArmIDs, "frozen arm")
	if err != nil {
		return nil, err
	}
	if len(arms) < 2 || len(arms) > maxModelPoolDiversityArms {
		return nil, fmt.Errorf("model-pool diversity reducer requires two through %d frozen arms", maxModelPoolDiversityArms)
	}
	cases, err := canonicalModelPoolIDs(input.PlannedCaseIDs, "planned case")
	if err != nil {
		return nil, err
	}
	if len(cases) == 0 || len(cases) > maxWorkerBrokerRequests/len(arms) {
		return nil, fmt.Errorf("model-pool dense matrix exceeds the broker request limit")
	}
	metricIDs := modelPoolMetricIDs(arms)
	if !input.Authoritative {
		return unavailableModelPoolMetrics(metricIDs, 0, map[modelPoolMissingReason]int{modelPoolNonAuthoritative: 1}), nil
	}

	armSet, caseSet := modelPoolIDSet(arms), modelPoolIDSet(cases)
	cells := make(map[string]map[string]modelPoolCell, len(cases))
	for _, caseID := range cases {
		cells[caseID] = make(map[string]modelPoolCell, len(arms))
	}
	for _, record := range input.PoolRecords {
		if record.TrackID != "model_pool" || record.ArmID == nil {
			return nil, fmt.Errorf("model-pool reducer received a record without one pool coordinate")
		}
		if !caseSet[record.CaseID] || !armSet[*record.ArmID] {
			return nil, fmt.Errorf("model-pool reducer received a record outside the frozen matrix")
		}
		if _, duplicate := cells[record.CaseID][*record.ArmID]; duplicate {
			return nil, fmt.Errorf("model-pool reducer received a duplicate case-arm coordinate")
		}
		cell, err := modelPoolCellFromRecord(record)
		if err != nil {
			return nil, err
		}
		cells[record.CaseID][*record.ArmID] = cell
	}

	support := summarizeModelPoolSupport(cases, arms, cells)

	metrics := make(map[string]modelPoolMetricEvidence, len(metricIDs))
	put := func(id string, value *float64, sampleCount int, reasons map[modelPoolMissingReason]int) {
		metrics[id] = modelPoolMetricEvidence{ID: id, Value: value, SampleCount: sampleCount, MissingReasonCounts: copyModelPoolReasons(reasons)}
	}
	put("model_pool.arm_count", modelPoolFloat(float64(len(arms))), len(cases), nil)
	put("model_pool.quality_shared_support_cases", modelPoolFloat(float64(len(support.qualityCompleteCases))), len(cases), support.qualityReasons)
	put("model_pool.quality_shared_support_fraction", modelPoolFloat(float64(len(support.qualityCompleteCases))/float64(len(cases))), len(cases), support.qualityReasons)
	put("model_pool.quality_cost_shared_support_cases", modelPoolFloat(float64(len(support.qualityCostCompleteCases))), len(cases), support.costReasons)
	put("model_pool.quality_cost_shared_support_fraction", modelPoolFloat(float64(len(support.qualityCostCompleteCases))/float64(len(cases))), len(cases), support.costReasons)
	putModelPoolQualityMetrics(put, cases, arms, cells, support)
	putModelPoolSuccessMetrics(put, cases, arms, cells, support)
	putModelPoolParetoMetrics(put, cases, arms, cells, support)

	if err := reduceModelPoolSelection(cases, arms, input.JointRecords, put); err != nil {
		return nil, err
	}
	result := make([]modelPoolMetricEvidence, 0, len(metrics))
	for _, id := range metricIDs {
		result = append(result, metrics[id])
	}
	return result, nil
}

func summarizeModelPoolSupport(cases, arms []string, cells map[string]map[string]modelPoolCell) modelPoolSupport {
	support := modelPoolSupport{
		qualityReasons:           make(map[modelPoolMissingReason]int),
		successReasons:           make(map[modelPoolMissingReason]int),
		costReasons:              make(map[modelPoolMissingReason]int),
		qualityCompleteCases:     make([]string, 0, len(cases)),
		qualityCostCompleteCases: make([]string, 0, len(cases)),
		successComplete:          true,
	}
	for _, caseID := range cases {
		qualityComplete, costComplete, allFailed := true, true, true
		for _, armID := range arms {
			cell, present := cells[caseID][armID]
			if !present {
				qualityComplete, costComplete, support.successComplete = false, false, false
				support.qualityReasons[modelPoolMissingArmCell]++
				support.costReasons[modelPoolMissingArmCell]++
				support.successReasons[modelPoolMissingArmCell]++
				continue
			}
			if !cell.successKnown {
				support.successComplete = false
				support.successReasons[modelPoolUnavailableRecord]++
			}
			if cell.quality == nil {
				qualityComplete, costComplete = false, false
				reason := modelPoolUnavailableRecord
				if cell.successKnown && cell.success {
					reason = modelPoolUngradedSuccess
				}
				support.qualityReasons[reason]++
				support.costReasons[reason]++
			}
			if cell.runtimeCost == nil {
				costComplete = false
				support.costReasons[modelPoolMissingRuntimeCost]++
			}
			if !cell.successKnown || cell.success {
				allFailed = false
			}
		}
		if qualityComplete {
			support.qualityCompleteCases = append(support.qualityCompleteCases, caseID)
		}
		if qualityComplete && costComplete {
			support.qualityCostCompleteCases = append(support.qualityCostCompleteCases, caseID)
		}
		if support.successComplete && allFailed {
			support.allArmFailures++
		}
	}
	return support
}

func putModelPoolQualityMetrics(put func(string, *float64, int, map[modelPoolMissingReason]int), cases, arms []string, cells map[string]map[string]modelPoolCell, support modelPoolSupport) {
	if len(support.qualityCompleteCases) != len(cases) {
		for _, armID := range arms {
			put(modelPoolArmMetricID(armID, "quality"), nil, len(support.qualityCompleteCases), support.qualityReasons)
			put(modelPoolArmMetricID(armID, "marginal_contribution"), nil, len(support.qualityCompleteCases), support.qualityReasons)
		}
		for _, id := range []string{"model_pool.best_single_quality", "model_pool.oracle_quality", "model_pool.oracle_gain", "model_pool.unique_wins", "model_pool.unique_win_rate", "model_pool.quality_dominated_arm_count"} {
			put(id, nil, len(support.qualityCompleteCases), support.qualityReasons)
		}
		return
	}
	qualityByArm, oracleValues, uniqueWins, marginal := reduceDensePoolQuality(cases, arms, cells)
	bestSingle := math.Inf(-1)
	for _, armID := range arms {
		if qualityByArm[armID] > bestSingle {
			bestSingle = qualityByArm[armID]
		}
		put(modelPoolArmMetricID(armID, "quality"), modelPoolFloat(qualityByArm[armID]), len(cases), nil)
		put(modelPoolArmMetricID(armID, "marginal_contribution"), modelPoolFloat(marginal[armID]), len(cases), nil)
	}
	oracle := modelPoolMean(oracleValues)
	put("model_pool.best_single_quality", modelPoolFloat(bestSingle), len(cases), nil)
	put("model_pool.oracle_quality", modelPoolFloat(oracle), len(cases), nil)
	put("model_pool.oracle_gain", modelPoolFloat(oracle-bestSingle), len(cases), nil)
	put("model_pool.unique_wins", modelPoolFloat(float64(uniqueWins)), len(cases), nil)
	put("model_pool.unique_win_rate", modelPoolFloat(float64(uniqueWins)/float64(len(cases))), len(cases), nil)
	put("model_pool.quality_dominated_arm_count", modelPoolFloat(float64(modelPoolQualityDominated(cases, arms, cells))), len(cases), nil)
}

func putModelPoolSuccessMetrics(put func(string, *float64, int, map[modelPoolMissingReason]int), cases, arms []string, cells map[string]map[string]modelPoolCell, support modelPoolSupport) {
	if !support.successComplete {
		for _, armID := range arms {
			put(modelPoolArmMetricID(armID, "success_rate"), nil, 0, support.successReasons)
		}
		for _, id := range []string{"model_pool.worst_arm_reliability", "model_pool.all_arm_failure_rate", "model_pool.mean_pairwise_failure_jaccard"} {
			put(id, nil, 0, support.successReasons)
		}
		return
	}
	for _, armID := range arms {
		successes := 0
		for _, caseID := range cases {
			if cells[caseID][armID].success {
				successes++
			}
		}
		put(modelPoolArmMetricID(armID, "success_rate"), modelPoolFloat(float64(successes)/float64(len(cases))), len(cases), nil)
	}
	put("model_pool.worst_arm_reliability", modelPoolFloat(modelPoolWorstReliability(cases, arms, cells)), len(cases), nil)
	put("model_pool.all_arm_failure_rate", modelPoolFloat(float64(support.allArmFailures)/float64(len(cases))), len(cases), nil)
	put("model_pool.mean_pairwise_failure_jaccard", modelPoolFloat(modelPoolFailureJaccard(cases, arms, cells)), len(cases), nil)
}

func putModelPoolParetoMetrics(put func(string, *float64, int, map[modelPoolMissingReason]int), cases, arms []string, cells map[string]map[string]modelPoolCell, support modelPoolSupport) {
	if len(support.qualityCompleteCases) == len(cases) && len(support.qualityCostCompleteCases) == len(cases) {
		paretoEvaluable, paretoDominated := modelPoolPareto(cases, arms, cells)
		put("model_pool.pareto_evaluable_arm_count", modelPoolFloat(float64(paretoEvaluable)), len(cases), nil)
		put("model_pool.pareto_dominated_arm_count", modelPoolFloat(float64(paretoDominated)), len(cases), nil)
		return
	}
	put("model_pool.pareto_evaluable_arm_count", nil, len(support.qualityCostCompleteCases), support.costReasons)
	put("model_pool.pareto_dominated_arm_count", nil, len(support.qualityCostCompleteCases), support.costReasons)
}

func canonicalModelPoolIDs(ids []string, label string) ([]string, error) {
	canonical := append([]string(nil), ids...)
	sort.Strings(canonical)
	for index, id := range canonical {
		if !evidenceIDPattern.MatchString(id) || (index > 0 && canonical[index-1] == id) {
			return nil, fmt.Errorf("model-pool %s identity is invalid", label)
		}
	}
	return canonical, nil
}

func modelPoolIDSet(ids []string) map[string]bool {
	set := make(map[string]bool, len(ids))
	for _, id := range ids {
		set[id] = true
	}
	return set
}

func modelPoolCellFromRecord(record executionRecordEvidence) (modelPoolCell, error) {
	cell := modelPoolCell{present: true, runtimeCost: record.RuntimeCost}
	if record.RuntimeCost != nil && (!finiteFloat(*record.RuntimeCost) || *record.RuntimeCost < 0) {
		return modelPoolCell{}, fmt.Errorf("model-pool runtime cost is invalid")
	}
	switch record.Status {
	case "failed":
		cell.successKnown, cell.success = true, false
		zero := 0.0
		cell.quality = &zero
	case "succeeded":
		if record.Success == nil {
			return modelPoolCell{}, fmt.Errorf("successful model-pool record omits success")
		}
		cell.successKnown, cell.success = true, *record.Success
		if !cell.success {
			zero := 0.0
			cell.quality = &zero
		} else if record.Quality != nil {
			if !finiteFloat(*record.Quality) || *record.Quality < 0 || *record.Quality > 1 {
				return modelPoolCell{}, fmt.Errorf("model-pool quality is invalid")
			}
			quality := *record.Quality
			cell.quality = &quality
		}
	case "unavailable":
		// An authoritative live matrix normally rejects this before reduction;
		// retaining it here produces a deterministic unavailable metric instead
		// of converting a missing observation into a successful zero.
	default:
		return modelPoolCell{}, fmt.Errorf("model-pool record status is invalid")
	}
	return cell, nil
}

func reduceDensePoolQuality(cases, arms []string, cells map[string]map[string]modelPoolCell) (map[string]float64, []float64, int, map[string]float64) {
	valuesByArm := make(map[string][]float64, len(arms))
	marginalValues := make(map[string][]float64, len(arms))
	oracles := make([]float64, 0, len(cases))
	uniqueWins := 0
	for _, caseID := range cases {
		best, secondBest, bestCount := densePoolCaseQuality(caseID, arms, cells, valuesByArm)
		oracles = append(oracles, best)
		if bestCount == 1 {
			uniqueWins++
		}
		for _, armID := range arms {
			without := best
			if *cells[caseID][armID].quality == best && bestCount == 1 {
				without = secondBest
			}
			marginalValues[armID] = append(marginalValues[armID], best-without)
		}
	}
	means, marginal := make(map[string]float64, len(arms)), make(map[string]float64, len(arms))
	for _, armID := range arms {
		means[armID] = modelPoolMean(valuesByArm[armID])
		marginal[armID] = modelPoolMean(marginalValues[armID])
	}
	return means, oracles, uniqueWins, marginal
}

func densePoolCaseQuality(
	caseID string,
	arms []string,
	cells map[string]map[string]modelPoolCell,
	valuesByArm map[string][]float64,
) (float64, float64, int) {
	best, secondBest, bestCount := math.Inf(-1), math.Inf(-1), 0
	for _, armID := range arms {
		quality := *cells[caseID][armID].quality
		valuesByArm[armID] = append(valuesByArm[armID], quality)
		switch {
		case quality > best:
			secondBest, best, bestCount = best, quality, 1
		case quality == best:
			bestCount++
		case quality > secondBest:
			secondBest = quality
		}
	}
	return best, secondBest, bestCount
}

func modelPoolQualityDominated(cases, arms []string, cells map[string]map[string]modelPoolCell) int {
	dominated := 0
	for _, candidate := range arms {
		for _, competitor := range arms {
			if candidate == competitor {
				continue
			}
			neverWorse, strictlyBetter := true, false
			for _, caseID := range cases {
				left, right := *cells[caseID][candidate].quality, *cells[caseID][competitor].quality
				if right < left {
					neverWorse = false
					break
				}
				strictlyBetter = strictlyBetter || right > left
			}
			if neverWorse && strictlyBetter {
				dominated++
				break
			}
		}
	}
	return dominated
}

func modelPoolWorstReliability(cases, arms []string, cells map[string]map[string]modelPoolCell) float64 {
	worst := 1.0
	for _, armID := range arms {
		successes := 0
		for _, caseID := range cases {
			if cells[caseID][armID].success {
				successes++
			}
		}
		reliability := float64(successes) / float64(len(cases))
		if reliability < worst {
			worst = reliability
		}
	}
	return worst
}

func modelPoolFailureJaccard(cases, arms []string, cells map[string]map[string]modelPoolCell) float64 {
	values := make([]float64, 0, len(arms)*(len(arms)-1)/2)
	for left := 0; left < len(arms); left++ {
		for right := left + 1; right < len(arms); right++ {
			intersection, union := 0, 0
			for _, caseID := range cases {
				leftFailed := !cells[caseID][arms[left]].success
				rightFailed := !cells[caseID][arms[right]].success
				if leftFailed || rightFailed {
					union++
				}
				if leftFailed && rightFailed {
					intersection++
				}
			}
			if union == 0 {
				values = append(values, 0)
			} else {
				values = append(values, float64(intersection)/float64(union))
			}
		}
	}
	return modelPoolMean(values)
}

func modelPoolPareto(cases, arms []string, cells map[string]map[string]modelPoolCell) (int, int) {
	quality, cost := make(map[string]float64, len(arms)), make(map[string]float64, len(arms))
	for _, armID := range arms {
		qualities, costs := make([]float64, 0, len(cases)), make([]float64, 0, len(cases))
		for _, caseID := range cases {
			qualities = append(qualities, *cells[caseID][armID].quality)
			costs = append(costs, *cells[caseID][armID].runtimeCost)
		}
		quality[armID], cost[armID] = modelPoolMean(qualities), modelPoolMean(costs)
	}
	dominated := 0
	for _, armID := range arms {
		for _, competitor := range arms {
			if armID == competitor {
				continue
			}
			if quality[competitor] >= quality[armID] && cost[competitor] <= cost[armID] &&
				(quality[competitor] > quality[armID] || cost[competitor] < cost[armID]) {
				dominated++
				break
			}
		}
	}
	return len(arms), dominated
}

func reduceModelPoolSelection(cases, arms []string, records []executionRecordEvidence, put func(string, *float64, int, map[modelPoolMissingReason]int)) error {
	armSet, caseSet := modelPoolIDSet(arms), modelPoolIDSet(cases)
	selected := make(map[string]string, len(cases))
	seen := make(map[string]struct{}, len(cases))
	reasons := make(map[modelPoolMissingReason]int)
	for _, record := range records {
		if record.TrackID != "joint" || !caseSet[record.CaseID] {
			return fmt.Errorf("model-pool reducer received joint evidence outside the planned matrix")
		}
		if _, duplicate := seen[record.CaseID]; duplicate {
			return fmt.Errorf("model-pool reducer received duplicate joint evidence")
		}
		seen[record.CaseID] = struct{}{}
		if record.Status == "unavailable" {
			if record.SelectedArmID != nil {
				return fmt.Errorf("unavailable joint evidence selects an arm")
			}
			reasons[modelPoolMissingSelection]++
			continue
		}
		if record.SelectedArmID == nil || !armSet[*record.SelectedArmID] {
			return fmt.Errorf("model-pool reducer received joint evidence with an invalid selected arm")
		}
		selected[record.CaseID] = *record.SelectedArmID
	}
	for _, caseID := range cases {
		if selected[caseID] == "" {
			reasons[modelPoolMissingSelection]++
		}
	}
	if len(reasons) != 0 {
		put("model_pool.selection_entropy_bits", nil, len(selected), reasons)
		put("model_pool.selection_arm_coverage", nil, len(selected), reasons)
		return nil
	}
	counts := make(map[string]int, len(arms))
	for _, armID := range selected {
		counts[armID]++
	}
	entropy := 0.0
	for _, armID := range arms {
		if count := counts[armID]; count != 0 {
			probability := float64(count) / float64(len(cases))
			entropy -= probability * math.Log2(probability)
		}
	}
	put("model_pool.selection_entropy_bits", modelPoolFloat(entropy), len(cases), nil)
	put("model_pool.selection_arm_coverage", modelPoolFloat(float64(len(counts))/float64(len(arms))), len(cases), nil)
	return nil
}

func modelPoolMetricIDs(arms []string) []string {
	ids := append([]string(nil), modelPoolStaticMetricIDs...)
	for _, armID := range arms {
		ids = append(ids,
			modelPoolArmMetricID(armID, "marginal_contribution"),
			modelPoolArmMetricID(armID, "quality"),
			modelPoolArmMetricID(armID, "success_rate"),
		)
	}
	sort.Strings(ids)
	return ids
}

func unavailableModelPoolMetrics(ids []string, sampleCount int, reasons map[modelPoolMissingReason]int) []modelPoolMetricEvidence {
	metrics := make([]modelPoolMetricEvidence, 0, len(ids))
	for _, id := range ids {
		metrics = append(metrics, modelPoolMetricEvidence{ID: id, SampleCount: sampleCount, MissingReasonCounts: copyModelPoolReasons(reasons)})
	}
	return metrics
}

func copyModelPoolReasons(reasons map[modelPoolMissingReason]int) map[modelPoolMissingReason]int {
	if len(reasons) == 0 {
		return nil
	}
	copy := make(map[modelPoolMissingReason]int, len(reasons))
	for reason, count := range reasons {
		if count > 0 {
			copy[reason] = count
		}
	}
	return copy
}

func modelPoolFloat(value float64) *float64 { return &value }

func modelPoolMean(values []float64) float64 {
	if len(values) == 0 {
		return 0
	}
	// Inputs are appended in canonical case/arm order. Compensated summation
	// makes that order stable across platforms without sorting by float value.
	sum, compensation := 0.0, 0.0
	for _, value := range values {
		adjusted := value - compensation
		next := sum + adjusted
		compensation = (next - sum) - adjusted
		sum = next
	}
	return sum / float64(len(values))
}
