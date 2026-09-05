package catalog

import (
	"fmt"
	"math"
	"strings"
)

func validateBenchmark(definition BenchmarkDefinition, path string) error {
	if strings.TrimSpace(definition.DisplayName) == "" || strings.TrimSpace(definition.Domain) == "" {
		return fmt.Errorf("%s.display_name and domain are required", path)
	}
	if len(definition.Metrics) == 0 {
		return fmt.Errorf("%s.metrics cannot be empty", path)
	}
	if strings.TrimSpace(definition.DefaultProfile) == "" || len(definition.Profiles) == 0 {
		return fmt.Errorf("%s.default_profile and profiles are required", path)
	}
	if err := validateBenchmarkProfiles(definition, path); err != nil {
		return err
	}
	return validateBenchmarkMetrics(definition.Metrics, path)
}

func validateBenchmarkProfiles(definition BenchmarkDefinition, path string) error {
	profiles := map[string]struct{}{}
	for index, profile := range definition.Profiles {
		profilePath := fmt.Sprintf("%s.profiles[%d]", path, index)
		if strings.TrimSpace(profile.ID) == "" || strings.TrimSpace(profile.DisplayName) == "" || strings.TrimSpace(profile.Description) == "" {
			return fmt.Errorf("%s is incomplete", profilePath)
		}
		if _, duplicate := profiles[profile.ID]; duplicate {
			return fmt.Errorf("%s.id %q is duplicated", profilePath, profile.ID)
		}
		profiles[profile.ID] = struct{}{}
	}
	if _, ok := profiles[definition.DefaultProfile]; !ok {
		return fmt.Errorf("%s.default_profile is not declared", path)
	}
	return nil
}

func validateBenchmarkMetrics(metrics []BenchmarkMetric, path string) error {
	seen := map[string]struct{}{}
	for index, metric := range metrics {
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
		profiles := map[string]struct{}{}
		for _, profile := range benchmark.Profiles {
			profiles[profile.ID] = struct{}{}
		}
		for _, metric := range benchmark.Metrics {
			metricID := id + "#" + metric.ID
			if _, exists := result[metricID]; exists {
				return nil, fmt.Errorf("benchmark metric %q is duplicated", metricID)
			}
			result[metricID] = metricDefinition{benchmark: id, metric: metric, domain: benchmark.Domain, profiles: profiles}
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
		if component.Benchmark == "" || component.BenchmarkProfile == "" {
			return fmt.Errorf("%s benchmark, metric, and benchmark_profile are required", path)
		}
		metricID := component.Benchmark + "#" + component.Metric
		metric, exists := metrics[metricID]
		if !exists {
			return fmt.Errorf("%s metric %q is unknown", path, metricID)
		}
		if _, ok := metric.profiles[component.BenchmarkProfile]; !ok {
			return fmt.Errorf("%s.benchmark_profile %q is unknown", path, component.BenchmarkProfile)
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
