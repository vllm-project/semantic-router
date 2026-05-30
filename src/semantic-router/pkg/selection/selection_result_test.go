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

package selection

import (
	"context"
	"errors"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

type stubSelector struct {
	result *SelectionResult
	err    error
}

func (s stubSelector) Select(ctx context.Context, selCtx *SelectionContext) (*SelectionResult, error) {
	return s.result, s.err
}

func (s stubSelector) Method() SelectionMethod {
	return SelectionMethod("stub")
}

func (s stubSelector) UpdateFeedback(ctx context.Context, feedback *Feedback) error {
	return nil
}

func (s stubSelector) Tier() AlgorithmTier {
	return TierSupported
}

func (s stubSelector) ExternalDependencies() []Dependency {
	return nil
}

func TestValidateSelectionResultRejectsInvalidContracts(t *testing.T) {
	selCtx := &SelectionContext{
		CandidateModels: []config.ModelRef{
			{Model: "model-a"},
			{Model: "model-b", LoRAName: "model-b-lora"},
		},
	}

	for _, tc := range []struct {
		name    string
		result  *SelectionResult
		wantErr error
	}{
		{name: "nil result", wantErr: ErrSelectionResultRequired},
		{name: "blank selected model", result: &SelectionResult{SelectedModel: " "}, wantErr: ErrSelectedModelRequired},
		{name: "not a candidate", result: &SelectionResult{SelectedModel: "other"}, wantErr: ErrSelectedModelNotCandidate},
	} {
		t.Run(tc.name, func(t *testing.T) {
			err := ValidateSelectionResult(selCtx, tc.result)
			if !errors.Is(err, tc.wantErr) {
				t.Fatalf("expected %v, got %v", tc.wantErr, err)
			}
		})
	}
}

func TestValidateSelectionResultAcceptsModelAndLoRAReferences(t *testing.T) {
	selCtx := &SelectionContext{
		CandidateModels: []config.ModelRef{
			{Model: "model-a"},
			{Model: "model-b", LoRAName: "model-b-lora"},
		},
	}

	for _, selected := range []string{"model-a", "model-b-lora"} {
		if err := ValidateSelectionResult(selCtx, &SelectionResult{SelectedModel: selected}); err != nil {
			t.Fatalf("expected selected model %q to be valid, got %v", selected, err)
		}
	}
}

func TestGlobalSelectRejectsInvalidSelectorResult(t *testing.T) {
	oldRegistry := GlobalRegistry
	defer func() {
		GlobalRegistry = oldRegistry
	}()

	method := SelectionMethod("stub_invalid_result")
	GlobalRegistry = NewRegistry()
	GlobalRegistry.Register(method, stubSelector{
		result: &SelectionResult{SelectedModel: "other"},
	})

	_, err := Select(context.Background(), method, &SelectionContext{
		CandidateModels: createCandidateModels("model-a", "model-b"),
	})
	if !errors.Is(err, ErrSelectedModelNotCandidate) {
		t.Fatalf("expected %v, got %v", ErrSelectedModelNotCandidate, err)
	}
}

func TestGlobalSelectRejectsNilSelectorResult(t *testing.T) {
	oldRegistry := GlobalRegistry
	defer func() {
		GlobalRegistry = oldRegistry
	}()

	method := SelectionMethod("stub_nil_result")
	GlobalRegistry = NewRegistry()
	GlobalRegistry.Register(method, stubSelector{})

	_, err := Select(context.Background(), method, &SelectionContext{
		CandidateModels: createCandidateModels("model-a"),
	})
	if !errors.Is(err, ErrSelectionResultRequired) {
		t.Fatalf("expected %v, got %v", ErrSelectionResultRequired, err)
	}
}
