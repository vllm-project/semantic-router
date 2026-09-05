package catalog

import (
	"fmt"
	"math"
	"sort"
	"strings"
)

type metricDefinition struct {
	benchmark string
	metric    BenchmarkMetric
	domain    string
	profiles  map[string]struct{}
}

type selectedEvaluationRecords struct {
	values   map[string]map[string]map[string]float64
	evidence map[string]map[string]map[string]string
	models   map[string]struct{}
	seen     map[string]struct{}
}

type indexValidationSummary struct {
	directDomainWeights map[string]float64
	hasNestedComponent  bool
}

type indexEvaluator struct {
	model    string
	effort   string
	indices  map[string]IndexDefinition
	metrics  map[string]metricDefinition
	values   map[string]float64
	evidence map[string]string
	memo     map[string]IndexResult
	visiting map[string]bool
}

type evaluatedIndexComponent struct {
	result     IndexComponentResult
	raw        float64
	present    bool
	domain     string
	provenance []string
}

type indexAccumulator struct {
	result         IndexResult
	weighted       float64
	domainWeighted map[string]float64
	domainCoverage map[string]float64
	provenance     map[string]struct{}
}

func (registry *Registry) compileEvaluations(
	input EvaluationConfig,
	cards map[string]EffectiveModelCard,
	reasoning map[string]ReasoningFamilyDefinition,
) (map[string]IndexDefinition, map[string]map[string]map[string]IndexResult, error) {
	benchmarks, err := registry.mergeBenchmarks(input.Benchmarks)
	if err != nil {
		return nil, nil, err
	}
	metrics, err := buildMetricDefinitions(benchmarks)
	if err != nil {
		return nil, nil, err
	}
	indices, err := registry.mergeIndices(input.Indices, metrics)
	if err != nil {
		return nil, nil, err
	}
	records := make([]EvaluationRecord, 0, len(registry.evaluations)+len(input.Records))
	for _, record := range registry.evaluations {
		card, exists := cards[record.Model]
		if exists && card.Provenance["id"] == SourceOperator {
			// A custom model may intentionally use the same user-facing name as a
			// built-in catalog identity. Never attach the built-in model's evidence
			// to that operator-owned card.
			continue
		}
		records = append(records, record)
	}
	records = append(records, input.Records...)
	selected, err := validateAndSelectRecords(records, metrics)
	if err != nil {
		return nil, nil, err
	}
	for model := range cards {
		selected.models[model] = struct{}{}
	}
	results, err := computeAllIndexResults(indices, metrics, cards, reasoning, selected)
	if err != nil {
		return nil, nil, err
	}
	return indices, results, nil
}

func (registry *Registry) mergeBenchmarks(inputs []BenchmarkDefinition) (map[string]BenchmarkDefinition, error) {
	benchmarks := make(map[string]BenchmarkDefinition, len(registry.benchmarks)+len(inputs))
	for id, definition := range registry.benchmarks {
		benchmarks[id] = definition
	}
	for index, definition := range inputs {
		path := fmt.Sprintf("evaluations.benchmarks[%d]", index)
		if strings.TrimSpace(definition.ID) == "" {
			return nil, fmt.Errorf("%s.id cannot be empty", path)
		}
		if _, exists := benchmarks[definition.ID]; exists {
			return nil, fmt.Errorf("%s.id %q conflicts with an existing benchmark", path, definition.ID)
		}
		if err := validateBenchmark(definition, path); err != nil {
			return nil, err
		}
		benchmarks[definition.ID] = definition
	}
	return benchmarks, nil
}

func (registry *Registry) mergeIndices(
	inputs []IndexDefinition,
	metrics map[string]metricDefinition,
) (map[string]IndexDefinition, error) {
	indices := make(map[string]IndexDefinition, len(registry.indices)+len(inputs))
	for id, definition := range registry.indices {
		indices[id] = definition
	}
	for index, definition := range inputs {
		path := fmt.Sprintf("evaluations.indices[%d]", index)
		if strings.TrimSpace(definition.ID) == "" {
			return nil, fmt.Errorf("%s.id cannot be empty", path)
		}
		if _, exists := indices[definition.ID]; exists {
			return nil, fmt.Errorf("%s.id %q conflicts with an existing index", path, definition.ID)
		}
		indices[definition.ID] = definition
	}
	if err := validateIndices(indices, metrics); err != nil {
		return nil, err
	}
	return indices, nil
}

func computeAllIndexResults(
	indices map[string]IndexDefinition,
	metrics map[string]metricDefinition,
	cards map[string]EffectiveModelCard,
	reasoning map[string]ReasoningFamilyDefinition,
	selected selectedEvaluationRecords,
) (map[string]map[string]map[string]IndexResult, error) {
	results := make(map[string]map[string]map[string]IndexResult, len(selected.models))
	for model := range selected.models {
		computed, err := computeModelIndexResults(model, indices, metrics, cards, reasoning, selected)
		if err != nil {
			return nil, err
		}
		results[model] = computed
	}
	return results, nil
}

func computeModelIndexResults(
	model string,
	indices map[string]IndexDefinition,
	metrics map[string]metricDefinition,
	cards map[string]EffectiveModelCard,
	reasoning map[string]ReasoningFamilyDefinition,
	selected selectedEvaluationRecords,
) (map[string]map[string]IndexResult, error) {
	results := map[string]map[string]IndexResult{}
	for _, effort := range modelReasoningEfforts(cards[model], reasoning, selected.values[model]) {
		evaluator := indexEvaluator{
			model: model, effort: effort, indices: indices, metrics: metrics,
			values: selected.values[model][effort], evidence: selected.evidence[model][effort],
			memo: map[string]IndexResult{}, visiting: map[string]bool{},
		}
		for _, indexID := range sortedKeys(indices) {
			result, err := evaluator.compute(indexID)
			if err != nil {
				return nil, err
			}
			evaluator.memo[indexID] = result
		}
		results[effort] = evaluator.memo
	}
	return results, nil
}

func modelReasoningEfforts(
	card EffectiveModelCard,
	reasoning map[string]ReasoningFamilyDefinition,
	measurements map[string]map[string]float64,
) []string {
	efforts := []string{"default"}
	if family, ok := reasoning[card.Card.ReasoningFamily]; ok {
		efforts = append([]string(nil), family.Levels...)
	}
	seen := make(map[string]struct{}, len(efforts))
	for _, effort := range efforts {
		seen[effort] = struct{}{}
	}
	for _, effort := range sortedKeys(measurements) {
		if _, ok := seen[effort]; ok {
			continue
		}
		efforts = append(efforts, effort)
	}
	return efforts
}

func preferredEvaluationEffort(
	card EffectiveModelCard,
	values map[string]map[string]IndexResult,
	reasoning map[string]ReasoningFamilyDefinition,
) string {
	if family, ok := reasoning[card.Card.ReasoningFamily]; ok {
		if _, present := values[family.Default]; present {
			return family.Default
		}
	}
	for _, effort := range []string{"default", "medium", "high", "max", "xhigh", "published"} {
		if _, ok := values[effort]; ok {
			return effort
		}
	}
	for _, effort := range sortedKeys(values) {
		return effort
	}
	return "default"
}

func validateAndSelectRecords(
	records []EvaluationRecord,
	metrics map[string]metricDefinition,
) (selectedEvaluationRecords, error) {
	selected := selectedEvaluationRecords{
		values:   map[string]map[string]map[string]float64{},
		evidence: map[string]map[string]map[string]string{},
		models:   map[string]struct{}{},
		seen:     map[string]struct{}{},
	}
	for index, record := range records {
		path := fmt.Sprintf("evaluations.records[%d]", index)
		if err := selected.add(record, path, metrics); err != nil {
			return selectedEvaluationRecords{}, err
		}
	}
	return selected, nil
}

func (selected *selectedEvaluationRecords) add(
	record EvaluationRecord,
	path string,
	metrics map[string]metricDefinition,
) error {
	if strings.TrimSpace(record.ID) == "" || strings.TrimSpace(record.Model) == "" {
		return fmt.Errorf("%s.id and model are required", path)
	}
	if _, duplicate := selected.seen[record.ID]; duplicate {
		return fmt.Errorf("%s.id %q is duplicated", path, record.ID)
	}
	selected.seen[record.ID] = struct{}{}
	selected.models[record.Model] = struct{}{}
	if record.Status != "available" {
		return nil
	}
	if err := validateAvailableRecord(record, path); err != nil {
		return err
	}
	selected.ensureModel(record.Model)
	for metricID, value := range record.Metrics {
		if err := selected.addMetric(record, metricID, value, path, metrics); err != nil {
			return err
		}
	}
	return nil
}

func validateAvailableRecord(record EvaluationRecord, path string) error {
	if strings.TrimSpace(record.Benchmark) == "" || strings.TrimSpace(record.BenchmarkProfile) == "" || strings.TrimSpace(record.ReasoningEffort) == "" {
		return fmt.Errorf("%s benchmark, benchmark_profile, and reasoning_effort are required", path)
	}
	if record.Evidence.Provenance == "" || record.Evidence.Verification == "" {
		return fmt.Errorf("%s.evidence provenance and verification are required", path)
	}
	if !record.Evidence.Redistributable {
		return fmt.Errorf("%s cannot publish non-redistributable evidence", path)
	}
	return nil
}

func (selected *selectedEvaluationRecords) ensureModel(model string) {
	if selected.values[model] != nil {
		return
	}
	selected.values[model] = map[string]map[string]float64{}
	selected.evidence[model] = map[string]map[string]string{}
}

func (selected *selectedEvaluationRecords) ensureEffort(model, effort string) {
	selected.ensureModel(model)
	if selected.values[model][effort] != nil {
		return
	}
	selected.values[model][effort] = map[string]float64{}
	selected.evidence[model][effort] = map[string]string{}
}

func (selected *selectedEvaluationRecords) addMetric(
	record EvaluationRecord,
	metricID string,
	value float64,
	path string,
	metrics map[string]metricDefinition,
) error {
	fullMetricID := record.Benchmark + "#" + metricID
	definition, exists := metrics[fullMetricID]
	if !exists {
		return fmt.Errorf("%s.metrics references unknown metric %q", path, fullMetricID)
	}
	if _, ok := definition.profiles[record.BenchmarkProfile]; !ok {
		return fmt.Errorf("%s.benchmark_profile %q is unknown", path, record.BenchmarkProfile)
	}
	if !finite(value) || value < definition.metric.Range[0] || value > definition.metric.Range[1] {
		return fmt.Errorf(
			"%s.metrics[%q] is outside [%g, %g]",
			path, metricID, definition.metric.Range[0], definition.metric.Range[1],
		)
	}
	selected.ensureEffort(record.Model, record.ReasoningEffort)
	key := evaluationMetricKey(record.Benchmark, record.BenchmarkProfile, metricID)
	if previous, exists := selected.values[record.Model][record.ReasoningEffort][key]; exists && previous != value {
		return fmt.Errorf(
			"%s conflicts with another available value for model %q effort %q metric %q",
			path, record.Model, record.ReasoningEffort, fullMetricID,
		)
	}
	selected.values[record.Model][record.ReasoningEffort][key] = value
	selected.evidence[record.Model][record.ReasoningEffort][key] = record.ID
	return nil
}

func evaluationMetricKey(benchmark, profile, metric string) string {
	return benchmark + "\x00" + profile + "\x00" + metric
}

func (evaluator *indexEvaluator) compute(indexID string) (IndexResult, error) {
	if result, ok := evaluator.memo[indexID]; ok {
		return result, nil
	}
	if evaluator.visiting[indexID] {
		return IndexResult{}, fmt.Errorf("index dependency cycle includes %q", indexID)
	}
	evaluator.visiting[indexID] = true
	defer delete(evaluator.visiting, indexID)
	definition := evaluator.indices[indexID]
	accumulator := newIndexAccumulator(evaluator.model, evaluator.effort, indexID, len(definition.Components))
	for _, component := range definition.Components {
		evaluated, err := evaluator.evaluateComponent(component)
		if err != nil {
			return IndexResult{}, fmt.Errorf("index[%s] component %q: %w", indexID, componentIdentity(component), err)
		}
		if err := accumulator.add(component, evaluated); err != nil {
			return IndexResult{}, fmt.Errorf("index[%s] component %q: %w", indexID, componentIdentity(component), err)
		}
	}
	return accumulator.finish(definition), nil
}

func (evaluator *indexEvaluator) evaluateComponent(component IndexComponent) (evaluatedIndexComponent, error) {
	result := IndexComponentResult{
		Benchmark: component.Benchmark, Metric: component.Metric,
		BenchmarkProfile: component.BenchmarkProfile, Index: component.Index,
		Weight: component.Weight, Status: "missing",
	}
	if component.Metric != "" {
		key := evaluationMetricKey(component.Benchmark, component.BenchmarkProfile, component.Metric)
		raw, present := evaluator.values[key]
		provenance := []string{}
		if recordID := evaluator.evidence[key]; recordID != "" {
			provenance = append(provenance, recordID)
			result.Evaluation = recordID
		}
		metricID := component.Benchmark + "#" + component.Metric
		return evaluatedIndexComponent{
			result: result, raw: raw, present: present,
			domain: evaluator.metrics[metricID].domain, provenance: provenance,
		}, nil
	}
	return evaluator.evaluateNestedComponent(component, result)
}

func (evaluator *indexEvaluator) evaluateNestedComponent(
	component IndexComponent,
	result IndexComponentResult,
) (evaluatedIndexComponent, error) {
	dependency, err := evaluator.compute(component.Index)
	if err != nil {
		return evaluatedIndexComponent{}, err
	}
	evaluator.memo[component.Index] = dependency
	evaluated := evaluatedIndexComponent{result: result, provenance: dependency.Provenance}
	if dependency.Score == nil || dependency.Status != "available" {
		return evaluated, nil
	}
	scale := evaluator.indices[component.Index].Scale
	evaluated.raw = (*dependency.Score - scale[0]) / (scale[1] - scale[0])
	evaluated.present = true
	return evaluated, nil
}

func newIndexAccumulator(model, effort, indexID string, componentCount int) indexAccumulator {
	return indexAccumulator{
		result: IndexResult{
			Model: model, ReasoningEffort: effort, Index: indexID, Status: "missing",
			Components: make([]IndexComponentResult, 0, componentCount),
			Domains:    map[string]float64{}, Provenance: []string{},
		},
		domainWeighted: map[string]float64{},
		domainCoverage: map[string]float64{},
		provenance:     map[string]struct{}{},
	}
}

func (accumulator *indexAccumulator) add(
	component IndexComponent,
	evaluated evaluatedIndexComponent,
) error {
	for _, recordID := range evaluated.provenance {
		accumulator.provenance[recordID] = struct{}{}
	}
	if evaluated.present {
		normalized, err := normalizeValue(evaluated.raw, component.Normalization)
		if err != nil {
			return err
		}
		evaluated.result.Status = "available"
		evaluated.result.Value = floatPointer(evaluated.raw)
		evaluated.result.Normalized = floatPointer(normalized)
		accumulator.result.Coverage += component.Weight
		accumulator.weighted += component.Weight * normalized
		accumulator.addDomain(evaluated.domain, component.Weight, normalized)
	}
	accumulator.result.Components = append(accumulator.result.Components, evaluated.result)
	return nil
}

func (accumulator *indexAccumulator) addDomain(domain string, weight, normalized float64) {
	if domain == "" {
		return
	}
	accumulator.domainCoverage[domain] += weight
	accumulator.domainWeighted[domain] += weight * normalized
}

func (accumulator *indexAccumulator) finish(definition IndexDefinition) IndexResult {
	coverage := accumulator.result.Coverage
	if indexAvailable(definition.Missing, coverage) {
		normalized := accumulator.weighted / coverage
		score := scaleValue(normalized, definition.Scale)
		accumulator.result.Status = "available"
		accumulator.result.Score = floatPointer(score)
		for domain, weighted := range accumulator.domainWeighted {
			accumulator.result.Domains[domain] = scaleValue(
				weighted/accumulator.domainCoverage[domain], definition.Scale,
			)
		}
	}
	for recordID := range accumulator.provenance {
		accumulator.result.Provenance = append(accumulator.result.Provenance, recordID)
	}
	sort.Strings(accumulator.result.Provenance)
	return accumulator.result
}

func indexAvailable(policy MissingPolicy, coverage float64) bool {
	switch policy.Policy {
	case "require_all":
		return math.Abs(coverage-1) <= 1e-9
	case "require_coverage":
		return coverage+1e-9 >= policy.Minimum
	case "reported_only":
		return coverage > 0
	default:
		return false
	}
}

func scaleValue(normalized float64, scale [2]float64) float64 {
	return scale[0] + normalized*(scale[1]-scale[0])
}

func normalizeValue(value float64, normalization Normalization) (float64, error) {
	kind := normalization.Type
	if kind == "" {
		kind = "identity"
	}
	switch kind {
	case "identity":
		return clamp01(value), nil
	case "one_minus":
		return clamp01(1 - value), nil
	case "linear_clamp":
		return clamp01((value - *normalization.Min) / (*normalization.Max - *normalization.Min)), nil
	case "piecewise_linear":
		return interpolatePiecewise(value, normalization.Points), nil
	case "logistic":
		return 1 / (1 + math.Exp(-*normalization.K*(value-*normalization.X0))), nil
	case "lookup":
		return lookupNormalizedValue(value, normalization.Values)
	default:
		return 0, fmt.Errorf("normalization %q is unsupported", kind)
	}
}

func interpolatePiecewise(value float64, points []NormalizationPoint) float64 {
	if value <= points[0].Input {
		return points[0].Output
	}
	for index := 1; index < len(points); index++ {
		if value <= points[index].Input {
			left, right := points[index-1], points[index]
			ratio := (value - left.Input) / (right.Input - left.Input)
			return left.Output + ratio*(right.Output-left.Output)
		}
	}
	return points[len(points)-1].Output
}

func lookupNormalizedValue(value float64, values map[string]float64) (float64, error) {
	key := fmt.Sprintf("%g", value)
	mapped, ok := values[key]
	if !ok {
		return 0, fmt.Errorf("lookup has no entry for %s", key)
	}
	return mapped, nil
}

func componentIdentity(component IndexComponent) string {
	if component.Metric != "" {
		return component.Benchmark + "#" + component.Metric + "@" + component.BenchmarkProfile
	}
	return component.Index
}

func clamp01(value float64) float64 {
	return math.Max(0, math.Min(1, value))
}

func finite(value float64) bool           { return !math.IsNaN(value) && !math.IsInf(value, 0) }
func floatPointer(value float64) *float64 { return &value }
