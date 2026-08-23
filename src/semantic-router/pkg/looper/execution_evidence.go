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

package looper

import "errors"

// ExecutionEvidence is the content-free accounting trace retained when a
// looper fails after one or more backend calls completed. It deliberately
// excludes response bodies so callers can account paid work without exposing
// candidate, judge, or verifier content.
type ExecutionEvidence struct {
	ModelsUsed []string
	Iterations int
	Usage      TokenUsage
}

// PartialExecutionError carries aggregate provider-reported usage across an
// algorithm failure boundary while preserving the original error chain.
type PartialExecutionError struct {
	cause    error
	evidence ExecutionEvidence
}

func (e *PartialExecutionError) Error() string {
	if e == nil || e.cause == nil {
		return "looper execution failed"
	}
	return e.cause.Error()
}

func (e *PartialExecutionError) Unwrap() error {
	if e == nil {
		return nil
	}
	return e.cause
}

// Evidence returns a defensive copy of the content-free execution trace.
func (e *PartialExecutionError) Evidence() ExecutionEvidence {
	if e == nil {
		return ExecutionEvidence{}
	}
	evidence := e.evidence
	evidence.ModelsUsed = append([]string(nil), evidence.ModelsUsed...)
	return evidence
}

// ExecutionEvidenceFromError extracts aggregate accounting from any looper
// algorithm without exposing its concrete failure implementation.
func ExecutionEvidenceFromError(err error) (ExecutionEvidence, bool) {
	var partial *PartialExecutionError
	if !errors.As(err, &partial) {
		return ExecutionEvidence{}, false
	}
	return partial.Evidence(), true
}

func newPartialExecutionError(cause error, evidence ExecutionEvidence) error {
	if cause == nil {
		return nil
	}
	if existing, ok := ExecutionEvidenceFromError(cause); ok {
		evidence = mergeExecutionEvidence(existing, evidence)
	}
	evidence.ModelsUsed = append([]string(nil), evidence.ModelsUsed...)
	return &PartialExecutionError{cause: cause, evidence: evidence}
}

func executionEvidenceFromResponses(
	responses []*ModelResponse,
	modelsUsed []string,
	iterations int,
) ExecutionEvidence {
	return ExecutionEvidence{
		ModelsUsed: append([]string(nil), modelsUsed...),
		Iterations: iterations,
		Usage:      SumUsage(responses...),
	}
}

func mergeExecutionEvidence(first, second ExecutionEvidence) ExecutionEvidence {
	models := append([]string(nil), first.ModelsUsed...)
	for _, model := range second.ModelsUsed {
		found := false
		for _, existing := range models {
			if existing == model {
				found = true
				break
			}
		}
		if !found && model != "" {
			models = append(models, model)
		}
	}
	return ExecutionEvidence{
		ModelsUsed: models,
		Iterations: first.Iterations + second.Iterations,
		Usage:      mergeTokenUsage(first.Usage, second.Usage),
	}
}
