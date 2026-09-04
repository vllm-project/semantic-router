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
}

type selectedEvaluationRecords struct {
	values   map[string]map[string]float64
	evidence map[string]map[string]string
	models   map[string]struct{}
	seen     map[string]struct{}
}

type indexValidationSummary struct {
	directDomainWeights map[string]float64
	hasNestedComponent  bool
}

type indexEvaluator struct {
	model    string
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
) (map[string]IndexDefinition, map[string]map[string]IndexResult, error) {
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
	results, err := computeAllIndexResults(indices, metrics, cards, selected)
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
	selected selectedEvaluationRecords,
) (map[string]map[string]IndexResult, error) {
	results := make(map[string]map[string]IndexResult, len(selected.models))
	for model := range selected.models {
		computed, err := computeModelIndexResults(model, indices, metrics, cards, selected)
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
	selected selectedEvaluationRecords,
) (map[string]IndexResult, error) {
	if card, ok := cards[model]; ok && card.Card.Kind == "virtual" {
		return notApplicableIndexResults(model, indices), nil
	}
	evaluator := indexEvaluator{
		model: model, indices: indices, metrics: metrics,
		values: selected.values[model], evidence: selected.evidence[model],
		memo: map[string]IndexResult{}, visiting: map[string]bool{},
	}
	for _, indexID := range sortedKeys(indices) {
		result, err := evaluator.compute(indexID)
		if err != nil {
			return nil, err
		}
		evaluator.memo[indexID] = result
	}
	return evaluator.memo, nil
}

func notApplicableIndexResults(model string, indices map[string]IndexDefinition) map[string]IndexResult {
	results := make(map[string]IndexResult, len(indices))
	for _, indexID := range sortedKeys(indices) {
		results[indexID] = IndexResult{
			Model: model, Index: indexID, Status: "not_applicable",
			Components: []IndexComponentResult{}, Provenance: []string{},
		}
	}
	return results
}

func validateBenchmark(definition BenchmarkDefinition, path string) error {
	if strings.TrimSpace(definition.DisplayName) == "" || strings.TrimSpace(definition.Domain) == "" {
		return fmt.Errorf("%s.display_name and domain are required", path)
	}
	if len(definition.Metrics) == 0 {
		return fmt.Errorf("%s.metrics cannot be empty", path)
	}
	seen := map[string]struct{}{}
	for index, metric := range definition.Metrics {
		metricPath := fmt.Sprintf("%s.metrics[%d]", path, index)
		if strings.TrimSpace(metric.ID) == "" {
			return fmt.Errorf("%s.id cannot be empty", metricPath)
		}
		if _, duplicate := seen[metric.ID]; duplicate {
			return fmt.Errorf("%s.id %q is duplicated", metricPath, metric.ID)
		}
		seen[metric.ID] = struct{}{}
		if metric.Direction != "higher_is_better" && metric.Direction != "lower_is_better" {
			return fmt.Errorf("%s.direction %q is unsupported", metricPath, metric.Direction)
		}
		if !finite(metric.Range[0]) || !finite(metric.Range[1]) || metric.Range[0] >= metric.Range[1] {
			return fmt.Errorf("%s.range is invalid", metricPath)
		}
	}
	return nil
}

func buildMetricDefinitions(benchmarks map[string]BenchmarkDefinition) (map[string]metricDefinition, error) {
	result := map[string]metricDefinition{}
	for id, benchmark := range benchmarks {
		if err := validateBenchmark(benchmark, "benchmark["+id+"]"); err != nil {
			return nil, err
		}
		for _, metric := range benchmark.Metrics {
			metricID := id + "#" + metric.ID
			if _, exists := result[metricID]; exists {
				return nil, fmt.Errorf("benchmark metric %q is duplicated", metricID)
			}
			result[metricID] = metricDefinition{benchmark: id, metric: metric, domain: benchmark.Domain}
		}
	}
	return result, nil
}

func validateIndices(indices map[string]IndexDefinition, metrics map[string]metricDefinition) error {
	for id, definition := range indices {
		if err := validateIndexDefinition(id, definition, indices, metrics); err != nil {
			return err
		}
	}
	return validateIndexDependencyGraph(indices)
}

func validateIndexDefinition(
	id string,
	definition IndexDefinition,
	indices map[string]IndexDefinition,
	metrics map[string]metricDefinition,
) error {
	path := "index[" + id + "]"
	if definition.Aggregation != "weighted_mean" {
		return fmt.Errorf("%s.aggregation %q is unsupported", path, definition.Aggregation)
	}
	if definition.Scale[0] >= definition.Scale[1] {
		return fmt.Errorf("%s.scale is invalid", path)
	}
	if err := validateMissingPolicy(definition.Missing, path+".missing"); err != nil {
		return err
	}
	if len(definition.Components) == 0 {
		return fmt.Errorf("%s.components cannot be empty", path)
	}
	if err := validateDomainWeights(definition.Domains, path); err != nil {
		return err
	}
	summary, err := validateIndexComponents(definition.Components, path, indices, metrics)
	if err != nil {
		return err
	}
	return validateDeclaredDomainWeights(definition.Domains, summary, path)
}

func validateMissingPolicy(policy MissingPolicy, path string) error {
	switch policy.Policy {
	case "require_all", "reported_only":
		return nil
	case "require_coverage":
		if policy.Minimum <= 0 || policy.Minimum > 1 {
			return fmt.Errorf("%s.minimum must be within (0, 1]", path)
		}
		return nil
	default:
		return fmt.Errorf("%s.policy %q is unsupported", path, policy.Policy)
	}
}

func validateDomainWeights(domains map[string]float64, path string) error {
	if len(domains) == 0 {
		return nil
	}
	total := 0.0
	for domain, weight := range domains {
		if strings.TrimSpace(domain) == "" || !finite(weight) || weight <= 0 {
			return fmt.Errorf("%s.domains[%q] must be positive and finite", path, domain)
		}
		total += weight
	}
	if math.Abs(total-1) > 1e-9 {
		return fmt.Errorf("%s domain weights sum to %.12g, want 1", path, total)
	}
	return nil
}

func validateIndexComponents(
	components []IndexComponent,
	path string,
	indices map[string]IndexDefinition,
	metrics map[string]metricDefinition,
) (indexValidationSummary, error) {
	summary := indexValidationSummary{directDomainWeights: map[string]float64{}}
	total := 0.0
	for index, component := range components {
		componentPath := fmt.Sprintf("%s.components[%d]", path, index)
		if err := validateIndexComponent(component, componentPath, indices, metrics, &summary); err != nil {
			return indexValidationSummary{}, err
		}
		total += component.Weight
	}
	if math.Abs(total-1) > 1e-9 {
		return indexValidationSummary{}, fmt.Errorf("%s component weights sum to %.12g, want 1", path, total)
	}
	return summary, nil
}

func validateIndexComponent(
	component IndexComponent,
	path string,
	indices map[string]IndexDefinition,
	metrics map[string]metricDefinition,
	summary *indexValidationSummary,
) error {
	if (component.Metric == "") == (component.Index == "") {
		return fmt.Errorf("%s must reference exactly one metric or index", path)
	}
	if !finite(component.Weight) || component.Weight <= 0 {
		return fmt.Errorf("%s.weight must be positive and finite", path)
	}
	if err := validateIndexComponentReference(component, path, indices, metrics, summary); err != nil {
		return err
	}
	return validateNormalization(component.Normalization, path+".normalization")
}

func validateIndexComponentReference(
	component IndexComponent,
	path string,
	indices map[string]IndexDefinition,
	metrics map[string]metricDefinition,
	summary *indexValidationSummary,
) error {
	if component.Metric != "" {
		metric, exists := metrics[component.Metric]
		if !exists {
			return fmt.Errorf("%s.metric %q is unknown", path, component.Metric)
		}
		summary.directDomainWeights[metric.domain] += component.Weight
		return nil
	}
	summary.hasNestedComponent = true
	if _, exists := indices[component.Index]; !exists {
		return fmt.Errorf("%s.index %q is unknown", path, component.Index)
	}
	return nil
}

func validateDeclaredDomainWeights(domains map[string]float64, summary indexValidationSummary, path string) error {
	if len(domains) == 0 || summary.hasNestedComponent {
		return nil
	}
	if len(summary.directDomainWeights) != len(domains) {
		return fmt.Errorf("%s domains do not match direct component weights", path)
	}
	for domain, weight := range domains {
		if math.Abs(summary.directDomainWeights[domain]-weight) > 1e-9 {
			return fmt.Errorf("%s domains do not match direct component weights", path)
		}
	}
	return nil
}

func validateIndexDependencyGraph(indices map[string]IndexDefinition) error {
	validator := indexDependencyValidator{
		indices: indices,
		visited: map[string]bool{},
		active:  map[string]bool{},
	}
	for id := range indices {
		if err := validator.visit(id); err != nil {
			return err
		}
	}
	return nil
}

type indexDependencyValidator struct {
	indices map[string]IndexDefinition
	visited map[string]bool
	active  map[string]bool
}

func (validator *indexDependencyValidator) visit(id string) error {
	if validator.active[id] {
		return fmt.Errorf("index dependency cycle includes %q", id)
	}
	if validator.visited[id] {
		return nil
	}
	validator.active[id] = true
	defer delete(validator.active, id)
	for _, component := range validator.indices[id].Components {
		if component.Index == "" {
			continue
		}
		if err := validator.visit(component.Index); err != nil {
			return err
		}
	}
	validator.visited[id] = true
	return nil
}

func validateNormalization(normalization Normalization, path string) error {
	kind := normalization.Type
	if kind == "" {
		kind = "identity"
	}
	switch kind {
	case "identity", "one_minus":
		return nil
	case "linear_clamp":
		return validateLinearClamp(normalization, path)
	case "piecewise_linear":
		return validatePiecewiseLinear(normalization.Points, path)
	case "logistic":
		return validateLogistic(normalization, path)
	case "lookup":
		return validateLookup(normalization.Values, path)
	default:
		return fmt.Errorf("%s.type %q is unsupported", path, kind)
	}
}

func validateLinearClamp(normalization Normalization, path string) error {
	if normalization.Min == nil || normalization.Max == nil || *normalization.Min >= *normalization.Max {
		return fmt.Errorf("%s linear_clamp requires min < max", path)
	}
	return nil
}

func validatePiecewiseLinear(points []NormalizationPoint, path string) error {
	if len(points) < 2 {
		return fmt.Errorf("%s piecewise_linear requires at least two points", path)
	}
	for index, point := range points {
		if !finite(point.Input) || !finite(point.Output) || point.Output < 0 || point.Output > 1 {
			return fmt.Errorf("%s.points[%d] is invalid", path, index)
		}
		if index > 0 && point.Input <= points[index-1].Input {
			return fmt.Errorf("%s.points inputs must be strictly increasing", path)
		}
	}
	return nil
}

func validateLogistic(normalization Normalization, path string) error {
	valid := normalization.K != nil && normalization.X0 != nil &&
		finite(*normalization.K) && finite(*normalization.X0) && *normalization.K != 0
	if !valid {
		return fmt.Errorf("%s logistic requires finite non-zero k and finite x0", path)
	}
	return nil
}

func validateLookup(values map[string]float64, path string) error {
	if len(values) == 0 {
		return fmt.Errorf("%s lookup requires values", path)
	}
	for key, value := range values {
		if strings.TrimSpace(key) == "" || !finite(value) || value < 0 || value > 1 {
			return fmt.Errorf("%s.values[%q] is invalid", path, key)
		}
	}
	return nil
}

func validateAndSelectRecords(
	records []EvaluationRecord,
	metrics map[string]metricDefinition,
) (selectedEvaluationRecords, error) {
	selected := selectedEvaluationRecords{
		values:   map[string]map[string]float64{},
		evidence: map[string]map[string]string{},
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
	selected.values[model] = map[string]float64{}
	selected.evidence[model] = map[string]string{}
}

func (selected *selectedEvaluationRecords) addMetric(
	record EvaluationRecord,
	metricID string,
	value float64,
	path string,
	metrics map[string]metricDefinition,
) error {
	definition, exists := metrics[metricID]
	if !exists {
		return fmt.Errorf("%s.metrics references unknown metric %q", path, metricID)
	}
	if !finite(value) || value < definition.metric.Range[0] || value > definition.metric.Range[1] {
		return fmt.Errorf(
			"%s.metrics[%q] is outside [%g, %g]",
			path, metricID, definition.metric.Range[0], definition.metric.Range[1],
		)
	}
	if previous, exists := selected.values[record.Model][metricID]; exists && previous != value {
		return fmt.Errorf(
			"%s conflicts with another available value for model %q metric %q",
			path, record.Model, metricID,
		)
	}
	selected.values[record.Model][metricID] = value
	selected.evidence[record.Model][metricID] = record.ID
	return nil
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
	accumulator := newIndexAccumulator(evaluator.model, indexID, len(definition.Components))
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
		Metric: component.Metric, Index: component.Index,
		Weight: component.Weight, Status: "missing",
	}
	if component.Metric != "" {
		raw, present := evaluator.values[component.Metric]
		provenance := []string{}
		if recordID := evaluator.evidence[component.Metric]; recordID != "" {
			provenance = append(provenance, recordID)
		}
		return evaluatedIndexComponent{
			result: result, raw: raw, present: present,
			domain: evaluator.metrics[component.Metric].domain, provenance: provenance,
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

func newIndexAccumulator(model, indexID string, componentCount int) indexAccumulator {
	return indexAccumulator{
		result: IndexResult{
			Model: model, Index: indexID, Status: "missing",
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
		return component.Metric
	}
	return component.Index
}

func clamp01(value float64) float64 {
	return math.Max(0, math.Min(1, value))
}

func finite(value float64) bool           { return !math.IsNaN(value) && !math.IsInf(value, 0) }
func floatPointer(value float64) *float64 { return &value }
