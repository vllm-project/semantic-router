package handlers

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"net/url"
	"regexp"
	"strings"

	modelcatalog "github.com/vllm-project/semantic-router/src/semantic-router/pkg/catalog"
)

var (
	errInvalidModelCatalogContract = errors.New("invalid model catalog contract")
	catalogHeaderName              = regexp.MustCompile("^[!#$%&'*+.^_`|~0-9A-Za-z-]+$")
)

// modelCatalogEnvelope is the complete sanitized read contract shared by the
// Dashboard's Add Model flow and the public model inventory. Resource types
// come from the Router catalog package so the HTTP boundary cannot drift from
// the runtime registry.
type modelCatalogEnvelope struct {
	SchemaVersion      string                                   `json:"schema_version"`
	Catalogs           []modelcatalog.CatalogHeader             `json:"catalogs"`
	Protocols          []modelcatalog.ProtocolDefinition        `json:"protocols"`
	Providers          []modelcatalog.ProviderDefinition        `json:"providers"`
	ReasoningFamilies  []modelcatalog.ReasoningFamilyDefinition `json:"reasoning_families"`
	Models             []modelcatalog.ModelCard                 `json:"models"`
	Benchmarks         []modelcatalog.BenchmarkDefinition       `json:"benchmarks"`
	Evaluations        []modelcatalog.EvaluationRecord          `json:"evaluations"`
	EvaluationCoverage []modelcatalog.EvaluationCoverage        `json:"evaluation_coverage"`
	Indices            []modelcatalog.IndexDefinition           `json:"indices"`
	IndexResults       []modelcatalog.IndexResult               `json:"index_results"`
	Configured         json.RawMessage                          `json:"configured,omitempty"`
}

func normalizeModelCatalogDocument(raw []byte) ([]byte, error) {
	var envelope modelCatalogEnvelope
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&envelope); err != nil {
		return nil, fmt.Errorf("%w: invalid JSON", errInvalidModelCatalogContract)
	}
	if err := decoder.Decode(&struct{}{}); !errors.Is(err, io.EOF) {
		return nil, fmt.Errorf("%w: trailing JSON", errInvalidModelCatalogContract)
	}
	if err := validateModelCatalogEnvelope(envelope); err != nil {
		return nil, fmt.Errorf("%w: %w", errInvalidModelCatalogContract, err)
	}

	// `configured` is interactive local-config state owned by the CLI. Never
	// expose paths, credentials, or other local state through the catalog API.
	envelope.Configured = nil
	return json.Marshal(envelope)
}

func validateModelCatalogEnvelope(envelope modelCatalogEnvelope) error {
	if envelope.SchemaVersion != "vllm-sr/model-catalog/v2" {
		return fmt.Errorf("unsupported schema version")
	}
	if len(envelope.Catalogs) == 0 || len(envelope.Protocols) == 0 ||
		len(envelope.Providers) == 0 || len(envelope.Models) == 0 ||
		len(envelope.Benchmarks) == 0 || len(envelope.Indices) == 0 {
		return fmt.Errorf("empty required inventory")
	}

	protocols, err := validateCatalogProtocols(envelope.Protocols)
	if err != nil {
		return err
	}
	_, err = validateCatalogProviders(envelope.Providers, protocols, envelope.Protocols)
	if err != nil {
		return err
	}
	reasoning, err := validateCatalogReasoning(envelope.ReasoningFamilies)
	if err != nil {
		return err
	}
	models, err := validateCatalogModels(envelope.Models, reasoning)
	if err != nil {
		return err
	}
	metrics, err := validateCatalogBenchmarks(envelope.Benchmarks)
	if err != nil {
		return err
	}
	indices, err := validateCatalogIndices(envelope.Indices, metrics)
	if err != nil {
		return err
	}
	if err := validateCatalogHeaders(envelope.Catalogs, models, indices); err != nil {
		return err
	}
	if err := validateCatalogProviderBindings(
		envelope.Providers,
		models,
		protocols,
		envelope.Models,
	); err != nil {
		return err
	}
	if err := validateCatalogEvaluations(envelope.Evaluations, models, metrics); err != nil {
		return err
	}
	if err := validateCatalogEvaluationCoverage(envelope.EvaluationCoverage, models, metrics); err != nil {
		return err
	}
	return validateCatalogIndexResults(envelope.IndexResults, models, indices)
}

func validateCatalogHeaders(
	headers []modelcatalog.CatalogHeader,
	models map[string]struct{},
	indices map[string]struct{},
) error {
	seen := make(map[string]struct{}, len(headers))
	for _, header := range headers {
		if header.CatalogVersion == "" ||
			(header.Channel != "latest" && header.Channel != "release") ||
			header.DefaultModel == "" || len(header.EnabledModels) == 0 ||
			header.DefaultIntelligenceIndex == "" {
			return fmt.Errorf("malformed catalog header")
		}
		key := header.Channel + ":" + header.CatalogVersion
		if _, exists := seen[key]; exists {
			return fmt.Errorf("duplicate catalog header")
		}
		seen[key] = struct{}{}
		if _, ok := models[header.DefaultModel]; !ok {
			return fmt.Errorf("catalog default model is unknown")
		}
		for _, model := range header.EnabledModels {
			if _, ok := models[model]; !ok {
				return fmt.Errorf("catalog enabled model is unknown")
			}
		}
		if _, ok := indices[header.DefaultIntelligenceIndex]; !ok {
			return fmt.Errorf("catalog default index is unknown")
		}
	}
	return nil
}

func validateCatalogProtocols(values []modelcatalog.ProtocolDefinition) (map[string]struct{}, error) {
	ids := make(map[string]struct{}, len(values))
	for _, protocol := range values {
		if protocol.ID == "" || protocol.DisplayName == "" || protocol.WireFormat == "" ||
			!strings.HasPrefix(protocol.DefaultBasePath, "/") ||
			len(protocol.Operations) == 0 || len(protocol.Capabilities) == 0 {
			return nil, fmt.Errorf("malformed protocol")
		}
		if _, exists := ids[protocol.ID]; exists {
			return nil, fmt.Errorf("duplicate protocol")
		}
		ids[protocol.ID] = struct{}{}
		operations := map[string]struct{}{}
		defaultBasePath := strings.TrimRight(protocol.DefaultBasePath, "/")
		for _, operation := range protocol.Operations {
			if operation.ID == "" || !strings.HasPrefix(operation.Path, "/") ||
				(operation.Path != defaultBasePath &&
					!strings.HasPrefix(operation.Path, defaultBasePath+"/")) ||
				(operation.Method != "GET" && operation.Method != "POST" && operation.Method != "DELETE") {
				return nil, fmt.Errorf("malformed protocol operation")
			}
			if _, exists := operations[operation.ID]; exists {
				return nil, fmt.Errorf("duplicate protocol operation")
			}
			operations[operation.ID] = struct{}{}
		}
	}
	return ids, nil
}

func validateCatalogProviders(
	values []modelcatalog.ProviderDefinition,
	protocols map[string]struct{},
	protocolDefinitions []modelcatalog.ProtocolDefinition,
) (map[string]struct{}, error) {
	operationKeys := make(map[string]struct{})
	for _, protocol := range protocolDefinitions {
		for _, operation := range protocol.Operations {
			operationKeys[protocol.ID+"#"+operation.ID] = struct{}{}
		}
	}
	ids := make(map[string]struct{}, len(values))
	for _, provider := range values {
		if provider.ID == "" || provider.DisplayName == "" || provider.Description == "" ||
			!oneOf(provider.Category, "start_here", "model_api", "private_runtime") ||
			!oneOf(provider.SupportTier, "native", "compatible", "runtime") ||
			len(provider.Protocols) == 0 || provider.DefaultProtocol == "" ||
			len(provider.SupportedOperations) == 0 ||
			provider.Presentation.Logo == "" || provider.Presentation.Monogram == "" ||
			!oneOf(provider.Auth.Strategy, "none", "bearer", "api_key_header") ||
			(provider.ReasoningTransport != "" && !oneOf(string(provider.ReasoningTransport), "chat_template_kwargs", "top_level_effort", "top_level_boolean", "reasoning_object", "thinking_object", "deepseek_thinking")) ||
			!oneOf(provider.Conformance.Status, "unverified", "fixture_verified", "live_verified") {
			return nil, fmt.Errorf("malformed provider")
		}
		if _, exists := ids[provider.ID]; exists {
			return nil, fmt.Errorf("duplicate provider")
		}
		ids[provider.ID] = struct{}{}
		defaultPresent := false
		providerProtocols := make(map[string]struct{}, len(provider.Protocols))
		for _, protocol := range provider.Protocols {
			if _, ok := protocols[protocol]; !ok {
				return nil, fmt.Errorf("provider protocol is unknown")
			}
			if _, duplicate := providerProtocols[protocol]; duplicate {
				return nil, fmt.Errorf("duplicate provider protocol")
			}
			providerProtocols[protocol] = struct{}{}
			defaultPresent = defaultPresent || protocol == provider.DefaultProtocol
		}
		supported := make(map[string]struct{}, len(provider.SupportedOperations))
		for _, operationKey := range provider.SupportedOperations {
			protocolID, _, ok := strings.Cut(operationKey, "#")
			if !ok {
				return nil, fmt.Errorf("malformed provider operation")
			}
			if _, known := operationKeys[operationKey]; !known {
				return nil, fmt.Errorf("provider operation is unknown")
			}
			if _, declared := providerProtocols[protocolID]; !declared {
				return nil, fmt.Errorf("provider operation uses an undeclared protocol")
			}
			if _, duplicate := supported[operationKey]; duplicate {
				return nil, fmt.Errorf("duplicate provider operation")
			}
			supported[operationKey] = struct{}{}
		}
		for protocolID := range providerProtocols {
			if _, ok := supported[protocolID+"#create"]; !ok {
				return nil, fmt.Errorf("provider protocol cannot create requests")
			}
		}
		for operationKey, operationPath := range provider.PathOverrides {
			if _, ok := supported[operationKey]; !ok || !strings.HasPrefix(operationPath, "/") {
				return nil, fmt.Errorf("provider path override is unsupported")
			}
		}
		if !defaultPresent || !validCatalogPresentationURL(provider.Presentation.Logo) ||
			(provider.DefaultBaseURL != "" && !validHTTPSURL(provider.DefaultBaseURL)) ||
			!validCatalogDefaultHeaders(provider) {
			return nil, fmt.Errorf("provider transport or presentation is unsafe")
		}
	}
	return ids, nil
}

func validCatalogDefaultHeaders(provider modelcatalog.ProviderDefinition) bool {
	forbidden := map[string]struct{}{
		"authorization":                       {},
		"proxy-authorization":                 {},
		"cookie":                              {},
		"set-cookie":                          {},
		strings.ToLower(provider.Auth.Header): {},
	}
	seen := make(map[string]struct{}, len(provider.DefaultHeaders))
	for name, value := range provider.DefaultHeaders {
		lower := strings.ToLower(name)
		if !catalogHeaderName.MatchString(name) || !validCatalogHeaderValue(value) {
			return false
		}
		if _, blocked := forbidden[lower]; blocked {
			return false
		}
		if _, duplicate := seen[lower]; duplicate {
			return false
		}
		seen[lower] = struct{}{}
	}
	return true
}

func validCatalogHeaderValue(value string) bool {
	if strings.TrimSpace(value) == "" {
		return false
	}
	for index := 0; index < len(value); index++ {
		if (value[index] < 0x20 && value[index] != '\t') || value[index] == 0x7f {
			return false
		}
	}
	return true
}

func validateCatalogReasoning(
	values []modelcatalog.ReasoningFamilyDefinition,
) (map[string]struct{}, error) {
	ids := make(map[string]struct{}, len(values))
	for _, family := range values {
		if family.ID == "" || family.Parameter == "" || len(family.Levels) == 0 ||
			!oneOf(family.Type, "chat_template_kwargs", "reasoning_effort", "top_level_reasoning_effort") ||
			!catalogContains(family.Levels, family.Default) {
			return nil, fmt.Errorf("malformed reasoning family")
		}
		if _, exists := ids[family.ID]; exists {
			return nil, fmt.Errorf("duplicate reasoning family")
		}
		ids[family.ID] = struct{}{}
	}
	return ids, nil
}

func validateCatalogModels(
	values []modelcatalog.ModelCard,
	reasoning map[string]struct{},
) (map[string]struct{}, error) {
	ids := make(map[string]struct{}, len(values))
	for _, model := range values {
		if model.ID == "" || model.DisplayName == "" || model.Description == "" ||
			!oneOf(model.Kind, "physical", "virtual") || model.Publisher == "" ||
			model.Presentation.Logo == "" || model.Presentation.Monogram == "" ||
			!validCatalogPresentationURL(model.Presentation.Logo) ||
			!oneOf(model.Distribution.Type, "proprietary_api", "open_weights", "router_recipe") ||
			!validHTTPSURL(model.Distribution.Source) ||
			(model.Distribution.Type == "open_weights" && model.Distribution.License == "") ||
			model.Family == "" ||
			!oneOf(model.Lifecycle, "experimental", "active", "deprecated", "removed") ||
			len(model.Capabilities) == 0 || len(model.Modalities.Input) == 0 ||
			len(model.Modalities.Output) == 0 ||
			model.Verification.Authority == "" ||
			!oneOf(model.Verification.Status, "claimed", "imported", "reproduced") ||
			(model.Verification.Source != "" && !validHTTPSURL(model.Verification.Source)) {
			return nil, fmt.Errorf("malformed model")
		}
		if _, exists := ids[model.ID]; exists {
			return nil, fmt.Errorf("duplicate model")
		}
		ids[model.ID] = struct{}{}
		if model.ReasoningFamily != "" {
			if _, ok := reasoning[model.ReasoningFamily]; !ok {
				return nil, fmt.Errorf("model reasoning family is unknown")
			}
		}
		if model.Kind == "virtual" &&
			(model.Generation < 1 || model.PolicyVersion == "" || model.Asset == "" ||
				model.Entrypoint == "" || model.Recipe == "" || len(model.Traits) == 0 ||
				model.Distribution.Type != "router_recipe" || !validModelCatalogRoles(model.Roles) ||
				!validModelCatalogDigest(model.Verification.AssetSHA256)) {
			return nil, fmt.Errorf("malformed virtual model")
		}
		if model.Kind == "physical" &&
			(model.Distribution.Type == "router_recipe" || model.Verification.Source == "") {
			return nil, fmt.Errorf("malformed physical model")
		}
	}
	return ids, nil
}

func validateCatalogProviderBindings(
	providers []modelcatalog.ProviderDefinition,
	models, protocols map[string]struct{},
	modelDefinitions []modelcatalog.ModelCard,
) error {
	boundModels := map[string]struct{}{}
	for _, provider := range providers {
		nativeIDs := map[string]struct{}{}
		pairs := map[string]struct{}{}
		for _, binding := range provider.Models {
			if binding.ID == "" || binding.Catalog == "" || len(binding.Protocols) == 0 ||
				!oneOf(binding.Lifecycle, "experimental", "active", "deprecated", "removed") ||
				!oneOf(binding.Verification.Status, "claimed", "imported", "reproduced") ||
				(binding.ReasoningTransport != "" && !oneOf(string(binding.ReasoningTransport), "chat_template_kwargs", "top_level_effort", "top_level_boolean", "reasoning_object", "thinking_object", "deepseek_thinking")) ||
				(binding.Verification.Source != "" && !validHTTPSURL(binding.Verification.Source)) {
				return fmt.Errorf("malformed provider catalog model")
			}
			if _, duplicate := nativeIDs[binding.ID]; duplicate {
				return fmt.Errorf("duplicate provider-native model id")
			}
			nativeIDs[binding.ID] = struct{}{}
			if _, ok := models[binding.Catalog]; !ok {
				return fmt.Errorf("provider catalog model is unknown")
			}
			boundModels[binding.Catalog] = struct{}{}
			for _, protocol := range binding.Protocols {
				if _, ok := protocols[protocol]; !ok || !catalogContains(provider.Protocols, protocol) {
					return fmt.Errorf("provider catalog model protocol is unknown")
				}
				pair := binding.Catalog + "#" + protocol
				if _, duplicate := pairs[pair]; duplicate {
					return fmt.Errorf("duplicate provider catalog model protocol")
				}
				pairs[pair] = struct{}{}
			}
		}
	}
	for _, model := range modelDefinitions {
		if model.Kind != "physical" || model.Lifecycle == "removed" {
			continue
		}
		if _, ok := boundModels[model.ID]; !ok {
			return fmt.Errorf("physical model has no provider binding")
		}
	}
	return nil
}

type catalogMetricContract struct {
	metric   modelcatalog.BenchmarkMetric
	profiles map[string]struct{}
}

func validateCatalogBenchmarks(
	values []modelcatalog.BenchmarkDefinition,
) (map[string]catalogMetricContract, error) {
	metrics := map[string]catalogMetricContract{}
	benchmarks := map[string]struct{}{}
	for _, benchmark := range values {
		if benchmark.ID == "" || benchmark.DisplayName == "" || benchmark.Domain == "" ||
			benchmark.DefaultProfile == "" || len(benchmark.Profiles) == 0 || len(benchmark.Metrics) == 0 ||
			(benchmark.Source != "" && !validHTTPSURL(benchmark.Source)) {
			return nil, fmt.Errorf("malformed benchmark")
		}
		if _, duplicate := benchmarks[benchmark.ID]; duplicate {
			return nil, fmt.Errorf("duplicate benchmark")
		}
		benchmarks[benchmark.ID] = struct{}{}
		profiles := map[string]struct{}{}
		for _, profile := range benchmark.Profiles {
			if profile.ID == "" || profile.DisplayName == "" || profile.Description == "" {
				return nil, fmt.Errorf("malformed benchmark profile")
			}
			if _, duplicate := profiles[profile.ID]; duplicate {
				return nil, fmt.Errorf("duplicate benchmark profile")
			}
			profiles[profile.ID] = struct{}{}
		}
		if _, ok := profiles[benchmark.DefaultProfile]; !ok {
			return nil, fmt.Errorf("benchmark default profile is unknown")
		}
		for _, metric := range benchmark.Metrics {
			key := benchmark.ID + "#" + metric.ID
			if metric.ID == "" || metric.Unit == "" ||
				!oneOf(metric.Direction, "higher_is_better", "lower_is_better") ||
				!finite(metric.Range[0]) || !finite(metric.Range[1]) || metric.Range[0] >= metric.Range[1] {
				return nil, fmt.Errorf("malformed benchmark metric")
			}
			if _, duplicate := metrics[key]; duplicate {
				return nil, fmt.Errorf("duplicate benchmark metric")
			}
			metrics[key] = catalogMetricContract{metric: metric, profiles: profiles}
		}
	}
	return metrics, nil
}

func validateCatalogEvaluations(
	values []modelcatalog.EvaluationRecord,
	models map[string]struct{},
	metrics map[string]catalogMetricContract,
) error {
	seen := map[string]struct{}{}
	availableMetrics := map[string]struct{}{}
	for _, evaluation := range values {
		if evaluation.ID == "" || evaluation.Benchmark == "" ||
			evaluation.BenchmarkProfile == "" || evaluation.ReasoningEffort == "" ||
			!oneOf(evaluation.Status, "available", "missing", "failed", "not_applicable", "withheld") ||
			!oneOf(evaluation.Evidence.Provenance, "vendor_claimed", "third_party", "vllm_sr_reproduced", "operator") ||
			!oneOf(evaluation.Evidence.Verification, "claimed", "imported", "reproduced") ||
			(evaluation.Evidence.Source != "" && !validHTTPSURL(evaluation.Evidence.Source)) {
			return fmt.Errorf("malformed evaluation")
		}
		if _, duplicate := seen[evaluation.ID]; duplicate {
			return fmt.Errorf("duplicate evaluation")
		}
		seen[evaluation.ID] = struct{}{}
		if _, ok := models[evaluation.Model]; !ok {
			return fmt.Errorf("evaluation model is unknown")
		}
		if evaluation.Status == "available" && (!evaluation.Evidence.Redistributable || len(evaluation.Metrics) == 0) {
			return fmt.Errorf("available evaluation lacks publishable evidence")
		}
		for metricID, value := range evaluation.Metrics {
			fullMetricID := evaluation.Benchmark + "#" + metricID
			contract, ok := metrics[fullMetricID]
			profileOK := false
			if ok {
				_, profileOK = contract.profiles[evaluation.BenchmarkProfile]
			}
			if !ok || !profileOK || !finite(value) || value < contract.metric.Range[0] || value > contract.metric.Range[1] {
				return fmt.Errorf("evaluation metric is invalid")
			}
			if evaluation.Status == "available" {
				key := evaluation.Model + "#" + evaluation.ReasoningEffort + "#" + fullMetricID + "#" + evaluation.BenchmarkProfile
				if _, duplicate := availableMetrics[key]; duplicate {
					return fmt.Errorf("evaluation metric has multiple available values")
				}
				availableMetrics[key] = struct{}{}
			}
		}
	}
	return nil
}

func validateCatalogEvaluationCoverage(
	values []modelcatalog.EvaluationCoverage,
	models map[string]struct{},
	metrics map[string]catalogMetricContract,
) error {
	seen := make(map[string]struct{}, len(values))
	for _, coverage := range values {
		metricID := coverage.Benchmark + "#" + coverage.Metric
		contract, metricOK := metrics[metricID]
		profileOK := false
		if metricOK {
			_, profileOK = contract.profiles[coverage.BenchmarkProfile]
		}
		_, modelOK := models[coverage.Model]
		key := coverage.Model + "#" + coverage.ReasoningEffort + "#" + metricID + "#" + coverage.BenchmarkProfile
		if _, duplicate := seen[key]; duplicate {
			return fmt.Errorf("duplicate evaluation coverage")
		}
		seen[key] = struct{}{}
		if !modelOK || coverage.ReasoningEffort == "" || !metricOK || !profileOK ||
			!oneOf(coverage.Status, "available", "missing", "failed", "not_applicable", "withheld") ||
			(coverage.Status == "available" && (coverage.Value == nil || coverage.Evaluation == "" || !finite(*coverage.Value))) ||
			(coverage.Status != "available" && (coverage.Value != nil || coverage.Evaluation != "")) {
			return fmt.Errorf("malformed evaluation coverage")
		}
		if coverage.Value != nil && (*coverage.Value < contract.metric.Range[0] || *coverage.Value > contract.metric.Range[1]) {
			return fmt.Errorf("evaluation coverage value is invalid")
		}
	}
	return nil
}

func validateCatalogIndices(
	values []modelcatalog.IndexDefinition,
	metrics map[string]catalogMetricContract,
) (map[string]struct{}, error) {
	ids := make(map[string]struct{}, len(values))
	for _, index := range values {
		if index.ID == "" || index.DisplayName == "" || index.Aggregation != "weighted_mean" ||
			index.Scale[0] >= index.Scale[1] || len(index.Components) == 0 ||
			!oneOf(index.Missing.Policy, "require_all", "require_coverage", "reported_only") ||
			(index.Methodology != "" && !validHTTPSURL(index.Methodology)) {
			return nil, fmt.Errorf("malformed index")
		}
		if _, duplicate := ids[index.ID]; duplicate {
			return nil, fmt.Errorf("duplicate index")
		}
		ids[index.ID] = struct{}{}
	}
	for _, index := range values {
		total := 0.0
		for _, component := range index.Components {
			if (component.Metric == "") == (component.Index == "") || component.Weight <= 0 || !finite(component.Weight) ||
				!oneOf(component.Normalization.Type, "identity", "one_minus", "linear_clamp", "piecewise_linear", "logistic", "lookup") {
				return nil, fmt.Errorf("malformed index component")
			}
			if component.Metric != "" {
				metric, ok := metrics[component.Benchmark+"#"+component.Metric]
				profileOK := false
				if ok {
					_, profileOK = metric.profiles[component.BenchmarkProfile]
				}
				if component.Benchmark == "" || component.BenchmarkProfile == "" || !ok || !profileOK {
					return nil, fmt.Errorf("index metric is unknown")
				}
			}
			if component.Index != "" {
				if _, ok := ids[component.Index]; !ok {
					return nil, fmt.Errorf("nested index is unknown")
				}
			}
			total += component.Weight
		}
		if math.Abs(total-1) > 1e-9 {
			return nil, fmt.Errorf("index weights do not sum to one")
		}
	}
	return ids, nil
}

func validateCatalogIndexResults(
	values []modelcatalog.IndexResult,
	models, indices map[string]struct{},
) error {
	seen := map[string]struct{}{}
	for _, result := range values {
		key := result.Model + "#" + result.ReasoningEffort + "#" + result.Index
		if _, duplicate := seen[key]; duplicate {
			return fmt.Errorf("duplicate index result")
		}
		seen[key] = struct{}{}
		_, modelOK := models[result.Model]
		_, indexOK := indices[result.Index]
		if !modelOK || result.ReasoningEffort == "" || !indexOK || !oneOf(result.Status, "available", "missing", "failed", "not_applicable", "withheld") ||
			!finite(result.Coverage) || result.Coverage < 0 || result.Coverage > 1 ||
			(result.Status == "available" && (result.Score == nil || !finite(*result.Score))) ||
			(result.Status != "available" && result.Score != nil) {
			return fmt.Errorf("malformed index result")
		}
	}
	return nil
}

func validModelCatalogDigest(value string) bool {
	if len(value) != len("sha256:")+64 || !strings.HasPrefix(value, "sha256:") {
		return false
	}
	for _, character := range value[len("sha256:"):] {
		if (character < '0' || character > '9') && (character < 'a' || character > 'f') {
			return false
		}
	}
	return true
}

func validModelCatalogRoles(roles []modelcatalog.ModelRole) bool {
	if len(roles) == 0 {
		return false
	}
	for _, role := range roles {
		if role.Name == "" || role.MinimumCandidates < 1 || len(role.Traits) == 0 ||
			len(role.RecommendedPool) < role.MinimumCandidates {
			return false
		}
	}
	return true
}

func validCatalogPresentationURL(value string) bool {
	if value == "monogram" || strings.HasPrefix(value, "package:") || strings.HasPrefix(value, "public:/") {
		return true
	}
	if strings.HasPrefix(value, "url:") {
		return validHTTPSURL(strings.TrimPrefix(value, "url:"))
	}
	return false
}

func validHTTPSURL(value string) bool {
	parsed, err := url.Parse(value)
	return err == nil && parsed.Scheme == "https" && parsed.Host != "" &&
		parsed.User == nil && parsed.Fragment == ""
}

func oneOf(value string, options ...string) bool {
	for _, option := range options {
		if value == option {
			return true
		}
	}
	return false
}

func catalogContains(values []string, expected string) bool {
	for _, value := range values {
		if value == expected {
			return true
		}
	}
	return false
}

func finite(value float64) bool {
	return !math.IsNaN(value) && !math.IsInf(value, 0)
}
