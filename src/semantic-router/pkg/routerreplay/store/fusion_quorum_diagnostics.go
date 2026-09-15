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

package store

// FusionQuorumDiagnostics captures content-free evidence for a Fusion panel
// that did not produce enough usable responses.
type FusionQuorumDiagnostics struct {
	RequiredCount int                             `json:"required_count"`
	UsableCount   int                             `json:"usable_count"`
	Attempts      []FusionPanelAttemptDiagnostics `json:"attempts,omitempty"`
}

// FusionPanelAttemptDiagnostics captures the terminal state and reported token
// usage for one Fusion panel call. It intentionally has no response or error
// text fields.
type FusionPanelAttemptDiagnostics struct {
	Model            string `json:"model"`
	State            string `json:"state"`
	PromptTokens     int64  `json:"prompt_tokens,omitempty"`
	CompletionTokens int64  `json:"completion_tokens,omitempty"`
	TotalTokens      int64  `json:"total_tokens,omitempty"`
}

func cloneFusionQuorumDiagnostics(value *FusionQuorumDiagnostics) *FusionQuorumDiagnostics {
	if value == nil {
		return nil
	}
	cloned := *value
	cloned.Attempts = append([]FusionPanelAttemptDiagnostics(nil), value.Attempts...)
	return &cloned
}
