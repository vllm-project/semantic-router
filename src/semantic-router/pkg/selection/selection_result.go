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
	"errors"
	"fmt"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

var (
	ErrSelectionResultRequired   = errors.New("selection result is required")
	ErrSelectedModelRequired     = errors.New("selected model is required")
	ErrSelectedModelNotCandidate = errors.New("selected model must reference a candidate model")
	// ErrNoEligibleCandidates identifies an intentional fail-closed policy
	// result. Callers must not turn it into a best-effort model fallback.
	ErrNoEligibleCandidates = errors.New("selection policy rejected all candidates")
)

// ValidateSelectionResult checks the common public selector output contract.
func ValidateSelectionResult(selCtx *SelectionContext, result *SelectionResult) error {
	_, err := ResolveSelectionCandidate(selCtx, result)
	return err
}

// ResolveSelectionCandidate is the single identity boundary for direct routing,
// composition and Eval. Model-only compatibility results must be unambiguous.
func ResolveSelectionCandidate(selCtx *SelectionContext, result *SelectionResult) (*config.ModelRef, error) {
	if err := ValidateSelectionContext(selCtx); err != nil {
		return nil, err
	}
	if result == nil {
		return nil, ErrSelectionResultRequired
	}
	selectedModel := strings.TrimSpace(result.SelectedModel)
	if selectedModel == "" {
		return nil, ErrSelectedModelRequired
	}
	if result.SelectedCandidate != nil {
		candidate := *result.SelectedCandidate
		if selectedModel != candidate.Model && (candidate.LoRAName == "" || selectedModel != candidate.LoRAName) {
			return nil, fmt.Errorf("%w: %q does not match selected candidate", ErrSelectedModelNotCandidate, result.SelectedModel)
		}
		for _, modelRef := range selCtx.CandidateModels {
			if CandidateIdentity(modelRef) == CandidateIdentity(candidate) {
				return result.WithCandidate(modelRef).SelectedCandidate, nil
			}
		}
		return nil, fmt.Errorf("%w: selected candidate is not declared", ErrSelectedModelNotCandidate)
	}
	if candidate := CandidateForModel(selCtx.CandidateModels, selectedModel, nil); candidate != nil {
		if result.Method != MethodPrompt || candidate.Model == selectedModel {
			return candidate, nil
		}
	}
	for _, ref := range selCtx.CandidateModels {
		if ref.Model == selectedModel || (result.Method != MethodPrompt && ref.LoRAName != "" && ref.LoRAName == selectedModel) {
			return nil, fmt.Errorf("%w: ambiguous candidate %q requires an exact ModelRef", ErrNoEligibleCandidates, selectedModel)
		}
	}
	return nil, fmt.Errorf("%w: %q", ErrSelectedModelNotCandidate, result.SelectedModel)
}
