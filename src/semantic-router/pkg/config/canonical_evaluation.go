/*
Copyright 2025 vLLM Semantic Router.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package config

import (
	"fmt"

	modelcatalog "github.com/vllm-project/semantic-router/src/semantic-router/pkg/catalog"
)

func canonicalEvaluationDefinitions(
	evaluation *CanonicalEvaluation,
	builtIn *modelcatalog.Registry,
	input *modelcatalog.EvaluationConfig,
) (map[string]modelcatalog.BenchmarkDefinition, error) {
	benchmarks := make(map[string]modelcatalog.BenchmarkDefinition)
	for _, definition := range builtIn.Benchmarks() {
		benchmarks[definition.ID] = definition
	}
	if evaluation == nil {
		return benchmarks, nil
	}

	cloned := cloneCanonicalEvaluation(evaluation)
	input.Benchmarks = cloned.Benchmarks
	input.Indices = cloned.Indices
	for index, definition := range input.Benchmarks {
		path := fmt.Sprintf("evaluation.benchmarks[%d].id", index)
		if !operatorResourceID.MatchString(definition.ID) {
			return nil, fmt.Errorf("%s must be a namespaced, versioned identity", path)
		}
		if _, exists := benchmarks[definition.ID]; exists {
			return nil, fmt.Errorf("%s %q conflicts with an existing benchmark", path, definition.ID)
		}
		benchmarks[definition.ID] = definition
	}

	indices := make(map[string]struct{}, len(builtIn.Indices())+len(input.Indices))
	for _, definition := range builtIn.Indices() {
		indices[definition.ID] = struct{}{}
	}
	for index, definition := range input.Indices {
		path := fmt.Sprintf("evaluation.indices[%d].id", index)
		if !operatorResourceID.MatchString(definition.ID) {
			return nil, fmt.Errorf("%s must be a namespaced, versioned identity", path)
		}
		if _, exists := indices[definition.ID]; exists {
			return nil, fmt.Errorf("%s %q conflicts with an existing index", path, definition.ID)
		}
		indices[definition.ID] = struct{}{}
	}
	return benchmarks, nil
}

func cloneCanonicalEvaluation(source *CanonicalEvaluation) *CanonicalEvaluation {
	if source == nil {
		return nil
	}
	result := &CanonicalEvaluation{
		Benchmarks: make([]modelcatalog.BenchmarkDefinition, len(source.Benchmarks)),
		Indices:    make([]modelcatalog.IndexDefinition, len(source.Indices)),
		Records:    cloneCanonicalEvaluationRecords(source.Records),
	}
	for index, definition := range source.Benchmarks {
		result.Benchmarks[index] = definition
		result.Benchmarks[index].Tags = append([]string(nil), definition.Tags...)
		result.Benchmarks[index].Profiles = append([]modelcatalog.BenchmarkProfile(nil), definition.Profiles...)
		result.Benchmarks[index].Metrics = make([]modelcatalog.BenchmarkMetric, len(definition.Metrics))
		for metricIndex, metric := range definition.Metrics {
			result.Benchmarks[index].Metrics[metricIndex] = metric
			if metric.Normalization != nil {
				normalization := cloneCatalogNormalization(*metric.Normalization)
				result.Benchmarks[index].Metrics[metricIndex].Normalization = &normalization
			}
		}
	}
	for index, definition := range source.Indices {
		result.Indices[index] = definition
		result.Indices[index].Domains = cloneFloatMap(definition.Domains)
		result.Indices[index].Components = make([]modelcatalog.IndexComponent, len(definition.Components))
		for componentIndex, component := range definition.Components {
			result.Indices[index].Components[componentIndex] = component
			result.Indices[index].Components[componentIndex].BenchmarkProfiles = append(
				[]string(nil), component.BenchmarkProfiles...,
			)
			result.Indices[index].Components[componentIndex].Normalization = cloneCatalogNormalization(component.Normalization)
		}
	}
	return result
}

func cloneCanonicalEvaluationRecords(source []CanonicalEvaluationRecord) []CanonicalEvaluationRecord {
	if len(source) == 0 {
		return nil
	}
	result := make([]CanonicalEvaluationRecord, len(source))
	for index, record := range source {
		result[index] = record
		result[index].Metrics = cloneFloatMap(record.Metrics)
		result[index].Metadata = cloneAnyMap(record.Metadata)
	}
	return result
}

func cloneCatalogNormalization(source modelcatalog.Normalization) modelcatalog.Normalization {
	result := source
	result.Points = append([]modelcatalog.NormalizationPoint(nil), source.Points...)
	result.Values = cloneFloatMap(source.Values)
	if source.Min != nil {
		value := *source.Min
		result.Min = &value
	}
	if source.Max != nil {
		value := *source.Max
		result.Max = &value
	}
	if source.K != nil {
		value := *source.K
		result.K = &value
	}
	if source.X0 != nil {
		value := *source.X0
		result.X0 = &value
	}
	return result
}

func cloneFloatMap(source map[string]float64) map[string]float64 {
	if source == nil {
		return nil
	}
	result := make(map[string]float64, len(source))
	for key, value := range source {
		result[key] = value
	}
	return result
}
