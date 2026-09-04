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
	SchemaVersion     string                                   `json:"schema_version"`
	Catalogs          []modelcatalog.CatalogHeader             `json:"catalogs"`
	Protocols         []modelcatalog.ProtocolDefinition        `json:"protocols"`
	Providers         []modelcatalog.ProviderDefinition        `json:"providers"`
	ReasoningFamilies []modelcatalog.ReasoningFamilyDefinition `json:"reasoning_families"`
	Models            []modelcatalog.ModelCard                 `json:"models"`
	Offerings         []modelcatalog.OfferingDefinition        `json:"offerings"`
	Benchmarks        []modelcatalog.BenchmarkDefinition       `json:"benchmarks"`
	Evaluations       []modelcatalog.EvaluationRecord          `json:"evaluations"`
	Indices           []modelcatalog.IndexDefinition           `json:"indices"`
	IndexResults      []modelcatalog.IndexResult               `json:"index_results"`
	Configured        json.RawMessage                          `json:"configured,omitempty"`
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
	providers, err := validateCatalogProviders(envelope.Providers, protocols, envelope.Protocols)
	if err != nil {
		return err
	}
	reasoning, err := validateCatalogReasoning(envelope.ReasoningFamilies)
	if err != nil {
		return err
	}
	models, err := validateCatalogModels(envelope.Models, protocols, reasoning)
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
	if err := validateCatalogOfferings(envelope.Offerings, providers, models, protocols); err != nil {
		return err
	}
	if err := validateCatalogEvaluations(envelope.Evaluations, models, metrics); err != nil {
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
			len(protocol.Operations) == 0 || len(protocol.Capabilities) == 0 {
			return nil, fmt.Errorf("malformed protocol")
		}
		if _, exists := ids[protocol.ID]; exists {
			return nil, fmt.Errorf("duplicate protocol")
		}
		ids[protocol.ID] = struct{}{}
		operations := map[string]struct{}{}
		for _, operation := range protocol.Operations {
			if operation.ID == "" || !strings.HasPrefix(operation.Path, "/") ||
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
			(provider.ReasoningTransport != "" && !oneOf(string(provider.ReasoningTransport), "chat_template_kwargs", "top_level_effort", "deepseek_thinking")) ||
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
	protocols map[string]struct{},
	reasoning map[string]struct{},
) (map[string]struct{}, error) {
	ids := make(map[string]struct{}, len(values))
	for _, model := range values {
		if model.ID == "" || model.DisplayName == "" || model.Description == "" ||
			!oneOf(model.Kind, "physical", "virtual") || model.Family == "" ||
			!oneOf(model.Lifecycle, "experimental", "active", "deprecated", "removed") ||
			len(model.Capabilities) == 0 || len(model.Modalities.Input) == 0 ||
			len(model.Modalities.Output) == 0 || len(model.Protocols) == 0 ||
			model.Verification.Authority == "" ||
			!oneOf(model.Verification.Status, "claimed", "imported", "reproduced") {
			return nil, fmt.Errorf("malformed model")
		}
		if _, exists := ids[model.ID]; exists {
			return nil, fmt.Errorf("duplicate model")
		}
		ids[model.ID] = struct{}{}
		for _, protocol := range model.Protocols {
			if _, ok := protocols[protocol]; !ok {
				return nil, fmt.Errorf("model protocol is unknown")
			}
		}
		if model.ReasoningFamily != "" {
			if _, ok := reasoning[model.ReasoningFamily]; !ok {
				return nil, fmt.Errorf("model reasoning family is unknown")
			}
		}
		if model.Kind == "virtual" &&
			(model.Generation < 1 || model.PolicyVersion == "" || model.Asset == "" ||
				model.Entrypoint == "" || model.Recipe == "" || len(model.Traits) == 0 ||
				!validModelCatalogRoles(model.Roles) || !validModelCatalogDigest(model.Verification.AssetSHA256)) {
			return nil, fmt.Errorf("malformed virtual model")
		}
	}
	return ids, nil
}

func validateCatalogOfferings(
	values []modelcatalog.OfferingDefinition,
	providers, models, protocols map[string]struct{},
) error {
	seen := map[string]struct{}{}
	for _, offering := range values {
		if offering.ID == "" || offering.ProviderModelID == "" || len(offering.Protocols) == 0 ||
			!oneOf(offering.Lifecycle, "experimental", "active", "deprecated", "removed") ||
			!oneOf(offering.Verification.Status, "claimed", "imported", "reproduced") {
			return fmt.Errorf("malformed offering")
		}
		if _, duplicate := seen[offering.ID]; duplicate {
			return fmt.Errorf("duplicate offering")
		}
		seen[offering.ID] = struct{}{}
		if _, ok := providers[offering.Provider]; !ok {
			return fmt.Errorf("offering provider is unknown")
		}
		if _, ok := models[offering.Model]; !ok {
			return fmt.Errorf("offering model is unknown")
		}
		for _, protocol := range offering.Protocols {
			if _, ok := protocols[protocol]; !ok {
				return fmt.Errorf("offering protocol is unknown")
			}
		}
	}
	return nil
}

func validateCatalogBenchmarks(
	values []modelcatalog.BenchmarkDefinition,
) (map[string]modelcatalog.BenchmarkMetric, error) {
	metrics := map[string]modelcatalog.BenchmarkMetric{}
	benchmarks := map[string]struct{}{}
	for _, benchmark := range values {
		if benchmark.ID == "" || benchmark.DisplayName == "" || benchmark.Domain == "" || len(benchmark.Metrics) == 0 ||
			(benchmark.Source != "" && !validHTTPSURL(benchmark.Source)) {
			return nil, fmt.Errorf("malformed benchmark")
		}
		if _, duplicate := benchmarks[benchmark.ID]; duplicate {
			return nil, fmt.Errorf("duplicate benchmark")
		}
		benchmarks[benchmark.ID] = struct{}{}
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
			metrics[key] = metric
		}
	}
	return metrics, nil
}

func validateCatalogEvaluations(
	values []modelcatalog.EvaluationRecord,
	models map[string]struct{},
	metrics map[string]modelcatalog.BenchmarkMetric,
) error {
	seen := map[string]struct{}{}
	for _, evaluation := range values {
		if evaluation.ID == "" || evaluation.MeasuredAt == "" ||
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
			metric, ok := metrics[metricID]
			if !ok || !finite(value) || value < metric.Range[0] || value > metric.Range[1] {
				return fmt.Errorf("evaluation metric is invalid")
			}
		}
	}
	return nil
}

func validateCatalogIndices(
	values []modelcatalog.IndexDefinition,
	metrics map[string]modelcatalog.BenchmarkMetric,
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
				if _, ok := metrics[component.Metric]; !ok {
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
		key := result.Model + "#" + result.Index
		if _, duplicate := seen[key]; duplicate {
			return fmt.Errorf("duplicate index result")
		}
		seen[key] = struct{}{}
		_, modelOK := models[result.Model]
		_, indexOK := indices[result.Index]
		if !modelOK || !indexOK || !oneOf(result.Status, "available", "missing", "failed", "not_applicable", "withheld") ||
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
