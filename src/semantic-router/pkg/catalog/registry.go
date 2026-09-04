package catalog

import (
	"encoding/json"
	"fmt"
	"sort"
	"sync"
)

// Registry is an immutable lookup view over one validated catalog snapshot.
// Lookup methods return values or defensive copies, never internal maps.
type Registry struct {
	header            CatalogHeader
	protocols         map[string]ProtocolDefinition
	providers         map[string]ProviderDefinition
	reasoningFamilies map[string]ReasoningFamilyDefinition
	models            map[string]ModelCard
	offerings         map[string]OfferingDefinition
	benchmarks        map[string]BenchmarkDefinition
	indices           map[string]IndexDefinition
	evaluations       []EvaluationRecord
	indexResults      map[string]map[string]IndexResult
	digest            string
}

var (
	builtInOnce sync.Once
	builtIn     *Registry
	builtInErr  error
)

// BuiltIn returns the release-embedded registry.
func BuiltIn() (*Registry, error) {
	builtInOnce.Do(func() {
		var document snapshot
		if err := json.Unmarshal([]byte(builtInCatalogJSON), &document); err != nil {
			builtInErr = fmt.Errorf("decode generated model catalog: %w", err)
			return
		}
		builtIn, builtInErr = registryFromSnapshot(document, builtInCatalogDigest)
	})
	return builtIn, builtInErr
}

func registryFromSnapshot(document snapshot, digest string) (*Registry, error) {
	if document.SchemaVersion != "vllm-sr/model-catalog/v2" {
		return nil, fmt.Errorf("unsupported model catalog schema %q", document.SchemaVersion)
	}
	if len(document.Catalogs) != 1 {
		return nil, fmt.Errorf("model catalog must contain exactly one active header")
	}
	registry := &Registry{
		header:            document.Catalogs[0],
		protocols:         make(map[string]ProtocolDefinition, len(document.Protocols)),
		providers:         make(map[string]ProviderDefinition, len(document.Providers)),
		reasoningFamilies: make(map[string]ReasoningFamilyDefinition, len(document.ReasoningFamilies)),
		models:            make(map[string]ModelCard, len(document.Models)),
		offerings:         make(map[string]OfferingDefinition, len(document.Offerings)),
		benchmarks:        make(map[string]BenchmarkDefinition, len(document.Benchmarks)),
		indices:           make(map[string]IndexDefinition, len(document.Indices)),
		evaluations:       append([]EvaluationRecord(nil), document.Evaluations...),
		indexResults:      make(map[string]map[string]IndexResult),
		digest:            digest,
	}
	for _, definition := range document.Protocols {
		registry.protocols[definition.ID] = definition
	}
	for _, definition := range document.Providers {
		registry.providers[definition.ID] = definition
	}
	for _, definition := range document.ReasoningFamilies {
		registry.reasoningFamilies[definition.ID] = definition
	}
	for _, definition := range document.Models {
		registry.models[definition.ID] = definition
	}
	for _, definition := range document.Offerings {
		registry.offerings[definition.ID] = definition
	}
	for _, definition := range document.Benchmarks {
		registry.benchmarks[definition.ID] = definition
	}
	for _, definition := range document.Indices {
		registry.indices[definition.ID] = definition
	}
	for _, result := range document.IndexResults {
		if registry.indexResults[result.Model] == nil {
			registry.indexResults[result.Model] = map[string]IndexResult{}
		}
		registry.indexResults[result.Model][result.Index] = result
	}
	return registry, nil
}

func (registry *Registry) Digest() string { return registry.digest }

func (registry *Registry) Header() CatalogHeader {
	header := registry.header
	header.EnabledModels = append([]string(nil), header.EnabledModels...)
	return header
}

func (registry *Registry) Protocol(id string) (ProtocolDefinition, bool) {
	value, ok := registry.protocols[id]
	return cloneProtocol(value), ok
}

func (registry *Registry) Provider(id string) (ProviderDefinition, bool) {
	value, ok := registry.providers[id]
	return cloneProvider(value), ok
}

func (registry *Registry) ReasoningFamily(id string) (ReasoningFamilyDefinition, bool) {
	value, ok := registry.reasoningFamilies[id]
	return cloneReasoningFamily(value), ok
}

func (registry *Registry) Model(id string) (ModelCard, bool) {
	value, ok := registry.models[id]
	return cloneModel(value), ok
}

func (registry *Registry) Index(id string) (IndexDefinition, bool) {
	value, ok := registry.indices[id]
	return cloneIndex(value), ok
}

func (registry *Registry) Offering(id string) (OfferingDefinition, bool) {
	value, ok := registry.offerings[id]
	value.Protocols = append([]string(nil), value.Protocols...)
	value.Restrictions = cloneMap(value.Restrictions)
	return value, ok
}

func (registry *Registry) Benchmark(id string) (BenchmarkDefinition, bool) {
	value, ok := registry.benchmarks[id]
	value.Metrics = append([]BenchmarkMetric(nil), value.Metrics...)
	return value, ok
}

func (registry *Registry) IndexResult(model, index string) (IndexResult, bool) {
	results := registry.indexResults[model]
	value, ok := results[index]
	return cloneIndexResult(value), ok
}

func (registry *Registry) ProviderIDs() []string  { return sortedKeys(registry.providers) }
func (registry *Registry) ModelIDs() []string     { return sortedKeys(registry.models) }
func (registry *Registry) ProtocolIDs() []string  { return sortedKeys(registry.protocols) }
func (registry *Registry) OfferingIDs() []string  { return sortedKeys(registry.offerings) }
func (registry *Registry) BenchmarkIDs() []string { return sortedKeys(registry.benchmarks) }
func (registry *Registry) IndexIDs() []string     { return sortedKeys(registry.indices) }

func (registry *Registry) Providers() []ProviderDefinition {
	ids := registry.ProviderIDs()
	result := make([]ProviderDefinition, 0, len(ids))
	for _, id := range ids {
		value, _ := registry.Provider(id)
		result = append(result, value)
	}
	return result
}

func (registry *Registry) Protocols() []ProtocolDefinition {
	ids := registry.ProtocolIDs()
	result := make([]ProtocolDefinition, 0, len(ids))
	for _, id := range ids {
		value, _ := registry.Protocol(id)
		result = append(result, value)
	}
	return result
}

func (registry *Registry) Models() []ModelCard {
	ids := registry.ModelIDs()
	result := make([]ModelCard, 0, len(ids))
	for _, id := range ids {
		value, _ := registry.Model(id)
		result = append(result, value)
	}
	return result
}

func (registry *Registry) Offerings() []OfferingDefinition {
	ids := registry.OfferingIDs()
	result := make([]OfferingDefinition, 0, len(ids))
	for _, id := range ids {
		value, _ := registry.Offering(id)
		result = append(result, value)
	}
	return result
}

func (registry *Registry) Benchmarks() []BenchmarkDefinition {
	ids := registry.BenchmarkIDs()
	result := make([]BenchmarkDefinition, 0, len(ids))
	for _, id := range ids {
		value, _ := registry.Benchmark(id)
		result = append(result, value)
	}
	return result
}

func (registry *Registry) Indices() []IndexDefinition {
	ids := registry.IndexIDs()
	result := make([]IndexDefinition, 0, len(ids))
	for _, id := range ids {
		value, _ := registry.Index(id)
		result = append(result, value)
	}
	return result
}

// Evaluations returns a defensive copy of the published evaluation evidence.
// Subject metadata is benchmark-specific and therefore requires a recursive
// copy instead of the shallow map copies used by closed catalog structures.
func (registry *Registry) Evaluations() []EvaluationRecord {
	result := make([]EvaluationRecord, len(registry.evaluations))
	for index, value := range registry.evaluations {
		result[index] = cloneEvaluation(value)
	}
	return result
}

func sortedKeys[Value any](values map[string]Value) []string {
	keys := make([]string, 0, len(values))
	for key := range values {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	return keys
}

func cloneProtocol(value ProtocolDefinition) ProtocolDefinition {
	value.Operations = append([]ProtocolOperation(nil), value.Operations...)
	value.Capabilities = append([]string(nil), value.Capabilities...)
	return value
}

func cloneProvider(value ProviderDefinition) ProviderDefinition {
	value.Protocols = append([]string(nil), value.Protocols...)
	value.SupportedOperations = append([]string(nil), value.SupportedOperations...)
	value.PathOverrides = cloneMap(value.PathOverrides)
	value.DefaultHeaders = cloneMap(value.DefaultHeaders)
	return value
}

func cloneReasoningFamily(value ReasoningFamilyDefinition) ReasoningFamilyDefinition {
	value.Levels = append([]string(nil), value.Levels...)
	return value
}

func cloneModel(value ModelCard) ModelCard {
	value.Capabilities = append([]string(nil), value.Capabilities...)
	value.Modalities.Input = append([]string(nil), value.Modalities.Input...)
	value.Modalities.Output = append([]string(nil), value.Modalities.Output...)
	value.Tags = append([]string(nil), value.Tags...)
	value.Protocols = append([]string(nil), value.Protocols...)
	value.Traits = append([]string(nil), value.Traits...)
	value.Roles = append([]ModelRole(nil), value.Roles...)
	for index := range value.Roles {
		value.Roles[index].Traits = append([]string(nil), value.Roles[index].Traits...)
		value.Roles[index].RecommendedPool = append([]string(nil), value.Roles[index].RecommendedPool...)
	}
	return value
}

func cloneIndex(value IndexDefinition) IndexDefinition {
	value.Domains = cloneMap(value.Domains)
	value.Components = append([]IndexComponent(nil), value.Components...)
	for index := range value.Components {
		value.Components[index].Normalization.Points = append([]NormalizationPoint(nil), value.Components[index].Normalization.Points...)
		value.Components[index].Normalization.Values = cloneMap(value.Components[index].Normalization.Values)
	}
	return value
}

func cloneIndexResult(value IndexResult) IndexResult {
	value.Components = append([]IndexComponentResult(nil), value.Components...)
	value.Domains = cloneMap(value.Domains)
	value.Provenance = append([]string(nil), value.Provenance...)
	return value
}

func cloneEvaluation(value EvaluationRecord) EvaluationRecord {
	value.Metrics = cloneMap(value.Metrics)
	if value.Subject != nil {
		subject := value.Subject
		value.Subject = make(EvaluationSubject, len(subject))
		for key, subjectValue := range subject {
			value.Subject[key] = cloneArbitraryValue(subjectValue)
		}
	}
	return value
}

func cloneArbitraryValue(value any) any {
	switch typed := value.(type) {
	case map[string]any:
		result := make(map[string]any, len(typed))
		for key, nested := range typed {
			result[key] = cloneArbitraryValue(nested)
		}
		return result
	case []any:
		result := make([]any, len(typed))
		for index, nested := range typed {
			result[index] = cloneArbitraryValue(nested)
		}
		return result
	default:
		return value
	}
}

func cloneMap[Value any](source map[string]Value) map[string]Value {
	if source == nil {
		return nil
	}
	result := make(map[string]Value, len(source))
	for key, value := range source {
		result[key] = value
	}
	return result
}
