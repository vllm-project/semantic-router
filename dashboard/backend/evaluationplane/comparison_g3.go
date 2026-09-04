package evaluationplane

const comparativeG3ReductionRef = "server-reduction:comparative-g3.v1"

func comparisonGates(
	baseline Report,
	candidate Report,
	statistics []ComparisonStatistic,
) []Gate {
	gates := append([]Gate(nil), candidate.Gates...)
	for index := range gates {
		if gates[index].ID == "G3" {
			gates[index] = reduceComparativeG3(gates[index], baseline, candidate, statistics)
		}
	}
	return gates
}

// reduceComparativeG3 keeps replay comparison useful without laundering a
// synthetic counterfactual into a release decision. Only the campaign's
// controlled AB/BA paired-live reducer is allowed to decide G3.
func reduceComparativeG3(
	gate Gate,
	baseline Report,
	candidate Report,
	statistics []ComparisonStatistic,
) Gate {
	gate.EvidenceRefs = []string{
		comparativeG3ReductionRef,
		"run:baseline:" + baseline.Run.ID,
		"run:candidate:" + candidate.Run.ID,
		"comparison-statistic:joint.normalized_regret",
	}
	gate.Observed, gate.Threshold, gate.SampleCount = nil, nil, nil
	gate.EvidenceLevel = "E0"
	gate.Owner = "recipe-and-model-pool"
	if gate.Disposition == GateDispositionNotApplicable {
		gate.Verdict = "not_applicable"
		gate.Rationale = "Synthetic comparative replay is not applicable to this change profile."
		return gate
	}
	gate.Verdict = "unavailable"
	if baseline.Run.Mode != ModeReplay || candidate.Run.Mode != ModeReplay {
		gate.Rationale = "A run comparison cannot decide G3; use a campaign with controlled AB/BA paired-live outcomes."
		return gate
	}
	statistic, found := comparisonStatisticByID(statistics, "joint.normalized_regret")
	if !found {
		gate.Rationale = "The synthetic replay diagnostic lacks complete pool-oracle and realized-quality records for normalized regret; G3 still requires paired-live outcomes."
		return gate
	}
	gate.SampleCount = intReference(statistic.SampleCount)
	gate.Rationale = "Server-reduced synthetic replay regret is retained as an E0 diagnostic only; it cannot pass or fail G3. A required G3 verdict needs controlled AB/BA paired-live outcomes."
	return gate
}

func comparisonStatisticByID(statistics []ComparisonStatistic, id string) (ComparisonStatistic, bool) {
	for _, statistic := range statistics {
		if statistic.ID == id {
			return statistic, true
		}
	}
	return ComparisonStatistic{}, false
}

func intReference(value int) *int { return &value }
