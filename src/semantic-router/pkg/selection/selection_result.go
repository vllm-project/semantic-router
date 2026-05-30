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
)

var (
	ErrSelectionResultRequired   = errors.New("selection result is required")
	ErrSelectedModelRequired     = errors.New("selected model is required")
	ErrSelectedModelNotCandidate = errors.New("selected model must reference a candidate model")
)

// ValidateSelectionResult checks the common public selector output contract.
func ValidateSelectionResult(selCtx *SelectionContext, result *SelectionResult) error {
	if err := ValidateSelectionContext(selCtx); err != nil {
		return err
	}
	if result == nil {
		return ErrSelectionResultRequired
	}
	selectedModel := strings.TrimSpace(result.SelectedModel)
	if selectedModel == "" {
		return ErrSelectedModelRequired
	}
	for _, modelRef := range selCtx.CandidateModels {
		if selectedModel == modelRef.Model || (modelRef.LoRAName != "" && selectedModel == modelRef.LoRAName) {
			return nil
		}
	}
	return fmt.Errorf("%w: %q", ErrSelectedModelNotCandidate, result.SelectedModel)
}
