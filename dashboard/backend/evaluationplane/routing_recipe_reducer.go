package evaluationplane

import (
	"fmt"
	"math"
	"sort"
)

func ReduceRoutingRecipeEvaluation(input RoutingRecipeReductionInput) (RoutingRecipeEvaluationReport, error) {
	decisions, err := validateRoutingRecipeReductionInput(input)
	if err != nil {
		return RoutingRecipeEvaluationReport{}, err
	}
	e1 := reduceRoutingRecipeE1(input.Plan, input.ExpectedCaseIDs, decisions)
	e2, err := reduceRoutingRecipeE2(input.Plan, input.ExpectedCaseIDs, decisions, input.Outcomes)
	if err != nil {
		return RoutingRecipeEvaluationReport{}, err
	}
	return RoutingRecipeEvaluationReport{
		ContractVersion: RoutingRecipeEvaluationContractVersion,
		PlanDigest:      input.Plan.PlanDigest,
		E1:              e1,
		E2:              e2,
	}, nil
}

func reduceRoutingRecipeE1(plan RoutingRecipePlan, cases []string, decisions map[string]RoutingRecipeDecisionSnapshot) RoutingRecipeE1Report {
	cases = sortedRoutingRecipeIDs(cases)
	report := RoutingRecipeE1Report{
		ExpectedDecisions: len(cases), ObservedDecisions: len(decisions),
		Signals:     reduceRoutingRecipeInputAvailability(plan.Signals, cases, decisions, false),
		Projections: reduceRoutingRecipeInputAvailability(projectionSpecsAsInputs(plan.Projections), cases, decisions, true),
	}
	for _, caseID := range cases {
		snapshot := decisions[caseID]
		eligibilityComplete := true
		for _, eligibility := range snapshot.Eligibility {
			if eligibility.State != "eligible" && eligibility.State != "ineligible" {
				eligibilityComplete = false
				break
			}
		}
		if eligibilityComplete {
			report.EligibilityComplete++
		}
		if snapshot.SelectedArmID != "" {
			for _, eligibility := range snapshot.Eligibility {
				if eligibility.ArmID == snapshot.SelectedArmID && eligibility.State == "eligible" {
					report.SelectedFeasible++
					break
				}
			}
		}
	}
	return report
}

func reduceRoutingRecipeInputAvailability(specs []RoutingRecipeInputSpec, cases []string, decisions map[string]RoutingRecipeDecisionSnapshot, projection bool) []RoutingRecipeInputAvailabilityReport {
	reports := make([]RoutingRecipeInputAvailabilityReport, 0, len(specs))
	for _, spec := range specs {
		report := RoutingRecipeInputAvailabilityReport{ID: spec.ID, Expected: len(cases)}
		latencies := make([]float64, 0, len(cases))
		for _, caseID := range cases {
			item, ok := routingRecipeObservedInput(decisions[caseID], spec.ID, projection)
			if !ok {
				continue
			}
			switch item.State {
			case "present":
				report.Present++
				if item.LatencyMS != nil {
					latencies = append(latencies, *item.LatencyMS)
				}
			case "missing":
				report.Missing++
			case "error":
				report.Error++
			case "timeout":
				report.Timeout++
			}
		}
		report.Latency = routingRecipeLatency(latencies)
		reports = append(reports, report)
	}
	sort.Slice(reports, func(left, right int) bool { return reports[left].ID < reports[right].ID })
	return reports
}

func routingRecipeObservedInput(
	decision RoutingRecipeDecisionSnapshot,
	specID string,
	projection bool,
) (RoutingRecipeObservedInput, bool) {
	observed := decision.Signals
	if projection {
		observed = decision.Projections
	}
	for _, item := range observed {
		if item.ID == specID {
			return item, true
		}
	}
	return RoutingRecipeObservedInput{}, false
}

func routingRecipeLatency(values []float64) RoutingRecipeLatencyReport {
	if len(values) < 2 {
		return RoutingRecipeLatencyReport{Available: false, Reason: "insufficient_latency_samples", SampleCount: len(values)}
	}
	sorted := append([]float64(nil), values...)
	sort.Float64s(sorted)
	return RoutingRecipeLatencyReport{
		Available: true, SampleCount: len(sorted),
		P50MS: routingRecipeQuantile(sorted, 0.5), P95MS: routingRecipeQuantile(sorted, 0.95),
	}
}

func reduceRoutingRecipeE2(plan RoutingRecipePlan, cases []string, decisions map[string]RoutingRecipeDecisionSnapshot, outcomes []RoutingRecipeOutcome) (RoutingRecipeE2Report, error) {
	cases = sortedRoutingRecipeIDs(cases)
	qualities, reason, err := routingRecipeEligibleQualities(plan, cases, decisions, outcomes)
	if err != nil {
		return RoutingRecipeE2Report{}, err
	}
	report := RoutingRecipeE2Report{ProjectionOutcomes: make([]RoutingRecipeProjectionOutcomeReport, 0, len(plan.Projections)), TopK: make([]RoutingRecipeTopKReport, 0, len(plan.TopK))}
	for _, projection := range plan.Projections {
		report.ProjectionOutcomes = append(report.ProjectionOutcomes, reduceRoutingRecipeProjectionOutcome(projection, cases, decisions, qualities, reason))
	}
	for _, k := range plan.TopK {
		report.TopK = append(report.TopK, reduceRoutingRecipeTopK(k, cases, decisions, qualities, reason))
	}
	report.OracleRegret = reduceRoutingRecipeRegret(cases, decisions, qualities, reason)
	sort.Slice(report.ProjectionOutcomes, func(left, right int) bool {
		return report.ProjectionOutcomes[left].ProjectionID < report.ProjectionOutcomes[right].ProjectionID
	})
	sort.Slice(report.TopK, func(left, right int) bool { return report.TopK[left].K < report.TopK[right].K })
	return report, nil
}

func routingRecipeEligibleQualities(plan RoutingRecipePlan, cases []string, decisions map[string]RoutingRecipeDecisionSnapshot, outcomes []RoutingRecipeOutcome) (map[string]map[string]float64, string, error) {
	byDecision := make(map[string]map[string]RoutingRecipeOutcome, len(cases))
	decisionByID := make(map[string]RoutingRecipeDecisionSnapshot, len(cases))
	for _, snapshot := range decisions {
		decisionByID[snapshot.DecisionID] = snapshot
	}
	for _, outcome := range outcomes {
		snapshot, found := decisionByID[outcome.DecisionID]
		if !found || outcome.CaseID != snapshot.CaseID || !validRoutingRecipeID(outcome.ArmID) || !finiteRoutingRecipeFloat(outcome.Quality) || outcome.Quality < 0 || outcome.Quality > 1 {
			return nil, "", fmt.Errorf("routing recipe outcome is not bound to a frozen decision")
		}
		if !outcome.ObservedAt.After(snapshot.ObservedAt) {
			return nil, "", fmt.Errorf("routing recipe outcome precedes or equals its server-observed decision")
		}
		if byDecision[outcome.DecisionID] == nil {
			byDecision[outcome.DecisionID] = make(map[string]RoutingRecipeOutcome)
		}
		if _, duplicate := byDecision[outcome.DecisionID][outcome.ArmID]; duplicate {
			return nil, "", fmt.Errorf("routing recipe outcome is duplicated")
		}
		byDecision[outcome.DecisionID][outcome.ArmID] = outcome
	}
	qualities := make(map[string]map[string]float64, len(cases))
	for _, caseID := range cases {
		snapshot := decisions[caseID]
		if snapshot.SelectionStatus != "selected" && snapshot.SelectionStatus != "fallback" {
			return nil, "selection_not_final", nil
		}
		eligible := make(map[string]struct{})
		for _, item := range snapshot.Eligibility {
			if item.State == "eligible" {
				eligible[item.ArmID] = struct{}{}
			}
		}
		if len(eligible) == 0 {
			return nil, "no_eligible_arms", nil
		}
		rows := byDecision[snapshot.DecisionID]
		if len(rows) != len(eligible) {
			return nil, "incomplete_eligible_pool", nil
		}
		qualities[caseID] = make(map[string]float64, len(eligible))
		for armID := range eligible {
			row, present := rows[armID]
			if !present {
				return nil, "incomplete_eligible_pool", nil
			}
			qualities[caseID][armID] = row.Quality
		}
		for armID := range rows {
			if _, eligibleArm := eligible[armID]; !eligibleArm {
				return nil, "", fmt.Errorf("routing recipe outcome exists for a non-eligible arm")
			}
		}
	}
	return qualities, "", nil
}

func reduceRoutingRecipeProjectionOutcome(spec RoutingRecipeProjectionSpec, cases []string, decisions map[string]RoutingRecipeDecisionSnapshot, qualities map[string]map[string]float64, prerequisiteReason string) RoutingRecipeProjectionOutcomeReport {
	report := RoutingRecipeProjectionOutcomeReport{ProjectionID: spec.ID}
	if prerequisiteReason != "" {
		unavailable := RoutingRecipeMetricAvailability{Reason: prerequisiteReason}
		report.Spearman, report.Brier, report.ECE10 = unavailable, unavailable, unavailable
		return report
	}
	predictions, outcomes, reason := routingRecipeProjectionPairs(spec, cases, decisions, qualities)
	if reason != "" {
		unavailable := RoutingRecipeMetricAvailability{Reason: reason}
		report.Spearman, report.Brier, report.ECE10 = unavailable, unavailable, unavailable
		return report
	}
	report.Spearman = routingRecipeSpearman(predictions, outcomes)
	if spec.ValueKind != "probability" || spec.OutcomeBinding != "selected_is_oracle" {
		report.Brier = RoutingRecipeMetricAvailability{Reason: "calibration_target_not_binary"}
		report.ECE10 = RoutingRecipeMetricAvailability{Reason: "calibration_target_not_binary"}
		return report
	}
	report.Brier, report.ECE10, report.Reliability = routingRecipeCalibration(predictions, outcomes)
	return report
}

func routingRecipeProjectionPairs(spec RoutingRecipeProjectionSpec, cases []string, decisions map[string]RoutingRecipeDecisionSnapshot, qualities map[string]map[string]float64) ([]float64, []float64, string) {
	predictions, outcomes := make([]float64, 0, len(cases)), make([]float64, 0, len(cases))
	for _, caseID := range cases {
		snapshot := decisions[caseID]
		var value *float64
		for _, projection := range snapshot.Projections {
			if projection.ID == spec.ID {
				value = projection.Value
				if projection.State != "present" {
					return nil, nil, "projection_not_present"
				}
				break
			}
		}
		if value == nil {
			return nil, nil, "projection_not_present"
		}
		selectedQuality, present := qualities[caseID][snapshot.SelectedArmID]
		if !present {
			return nil, nil, "selected_outcome_unavailable"
		}
		outcome := selectedQuality
		if spec.OutcomeBinding == "selected_is_oracle" {
			maximum := routingRecipeMaximum(qualities[caseID])
			outcome = 0
			if selectedQuality == maximum {
				outcome = 1
			}
		}
		predictions, outcomes = append(predictions, *value), append(outcomes, outcome)
	}
	return predictions, outcomes, ""
}

func reduceRoutingRecipeTopK(k int, cases []string, decisions map[string]RoutingRecipeDecisionSnapshot, qualities map[string]map[string]float64, prerequisiteReason string) RoutingRecipeTopKReport {
	report := RoutingRecipeTopKReport{K: k}
	if prerequisiteReason != "" {
		report.Recall = RoutingRecipeMetricAvailability{Reason: prerequisiteReason}
		return report
	}
	hits := 0
	for _, caseID := range cases {
		snapshot := decisions[caseID]
		if len(snapshot.RankedArmIDs) < k {
			report.Recall = RoutingRecipeMetricAvailability{Reason: "ranking_shorter_than_k"}
			return report
		}
		maximum := routingRecipeMaximum(qualities[caseID])
		for _, armID := range snapshot.RankedArmIDs[:k] {
			if qualities[caseID][armID] == maximum {
				hits++
				break
			}
		}
	}
	report.Recall = RoutingRecipeMetricAvailability{Available: true, Value: float64(hits) / float64(len(cases)), SampleCount: len(cases)}
	return report
}

func reduceRoutingRecipeRegret(cases []string, decisions map[string]RoutingRecipeDecisionSnapshot, qualities map[string]map[string]float64, prerequisiteReason string) RoutingRecipeMetricAvailability {
	if prerequisiteReason != "" {
		return RoutingRecipeMetricAvailability{Reason: prerequisiteReason}
	}
	values := make([]float64, 0, len(cases))
	for _, caseID := range cases {
		snapshot := decisions[caseID]
		selected, present := qualities[caseID][snapshot.SelectedArmID]
		if !present {
			return RoutingRecipeMetricAvailability{Reason: "selected_outcome_unavailable"}
		}
		values = append(values, routingRecipeMaximum(qualities[caseID])-selected)
	}
	return RoutingRecipeMetricAvailability{Available: true, Value: routingRecipeCompensatedSum(values) / float64(len(cases)), SampleCount: len(cases)}
}

func routingRecipeSpearman(predictions, outcomes []float64) RoutingRecipeMetricAvailability {
	if len(predictions) < 3 {
		return RoutingRecipeMetricAvailability{Reason: "insufficient_pairs", SampleCount: len(predictions)}
	}
	x, y := routingRecipeRanks(predictions), routingRecipeRanks(outcomes)
	meanX, meanY := routingRecipeMean(x), routingRecipeMean(y)
	varianceX, varianceY, covariance := 0.0, 0.0, 0.0
	for index := range x {
		deltaX, deltaY := x[index]-meanX, y[index]-meanY
		varianceX += deltaX * deltaX
		varianceY += deltaY * deltaY
		covariance += deltaX * deltaY
	}
	if varianceX == 0 || varianceY == 0 {
		return RoutingRecipeMetricAvailability{Reason: "insufficient_nonconstant_pairs", SampleCount: len(predictions)}
	}
	value := covariance / math.Sqrt(varianceX*varianceY)
	return RoutingRecipeMetricAvailability{Available: true, Value: value, SampleCount: len(predictions)}
}

func routingRecipeCalibration(predictions, outcomes []float64) (RoutingRecipeMetricAvailability, RoutingRecipeMetricAvailability, []RoutingRecipeReliabilityBin) {
	bins := make([]RoutingRecipeReliabilityBin, 10)
	predictionTotals, outcomeTotals := make([]float64, 10), make([]float64, 10)
	for index, prediction := range predictions {
		bin := int(prediction * 10)
		if bin == 10 {
			bin = 9
		}
		bins[bin].Count++
		predictionTotals[bin] += prediction
		outcomeTotals[bin] += outcomes[index]
		// Accumulate below with a compensated, canonical-order sum.
	}
	eceTerms := make([]float64, 0, len(bins))
	for index := range bins {
		bins[index].Lower, bins[index].Upper = float64(index)/10, float64(index+1)/10
		if bins[index].Count == 0 {
			continue
		}
		bins[index].MeanPrediction = predictionTotals[index] / float64(bins[index].Count)
		bins[index].ObservedFrequency = outcomeTotals[index] / float64(bins[index].Count)
		eceTerms = append(eceTerms, float64(bins[index].Count)/float64(len(predictions))*math.Abs(bins[index].MeanPrediction-bins[index].ObservedFrequency))
	}
	brierTerms := make([]float64, len(predictions))
	for index, prediction := range predictions {
		brierTerms[index] = (prediction - outcomes[index]) * (prediction - outcomes[index])
	}
	return RoutingRecipeMetricAvailability{Available: true, Value: routingRecipeCompensatedSum(brierTerms) / float64(len(predictions)), SampleCount: len(predictions)},
		RoutingRecipeMetricAvailability{Available: true, Value: routingRecipeCompensatedSum(eceTerms), SampleCount: len(predictions)}, bins
}

func routingRecipeRanks(values []float64) []float64 {
	type item struct {
		value float64
		index int
	}
	items := make([]item, len(values))
	for index, value := range values {
		items[index] = item{value: value, index: index}
	}
	sort.Slice(items, func(left, right int) bool { return items[left].value < items[right].value })
	ranks := make([]float64, len(values))
	for start := 0; start < len(items); {
		end := start + 1
		for end < len(items) && items[end].value == items[start].value {
			end++
		}
		rank := (float64(start+1) + float64(end)) / 2
		for index := start; index < end; index++ {
			ranks[items[index].index] = rank
		}
		start = end
	}
	return ranks
}

func routingRecipeQuantile(sorted []float64, quantile float64) float64 {
	position := quantile * float64(len(sorted)-1)
	lower := int(math.Floor(position))
	upper := int(math.Ceil(position))
	return sorted[lower] + (position-float64(lower))*(sorted[upper]-sorted[lower])
}

func routingRecipeMaximum(values map[string]float64) float64 {
	maximum := -1.0
	for _, value := range values {
		if value > maximum {
			maximum = value
		}
	}
	return maximum
}

func routingRecipeMean(values []float64) float64 {
	return routingRecipeCompensatedSum(values) / float64(len(values))
}

// routingRecipeCompensatedSum gives deterministic, permutation-independent
// accumulation once callers have put observations in their canonical order.
func routingRecipeCompensatedSum(values []float64) float64 {
	sum, correction := 0.0, 0.0
	for _, value := range values {
		t := sum + value
		if math.Abs(sum) >= math.Abs(value) {
			correction += (sum - t) + value
		} else {
			correction += (value - t) + sum
		}
		sum = t
	}
	return sum + correction
}
