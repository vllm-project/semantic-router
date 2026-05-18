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
	"time"
)

// OfflineDatasetRecord is a single training observation for offline weight updates.
// It captures the routing decision, outcome, and session context from a completed
// request so that an offline trainer can replay and learn from historical evidence.
type OfflineDatasetRecord struct {
	// SessionID groups records into multi-turn conversations.
	SessionID string `json:"session_id"`

	// TurnIndex is the zero-based position of this turn within the session.
	TurnIndex int `json:"turn_index"`

	// DecisionName is the matched routing decision (category context).
	DecisionName string `json:"decision_name,omitempty"`

	// CategoryName is the detected domain category (e.g. "math", "coding").
	CategoryName string `json:"category_name,omitempty"`

	// UserID identifies the user for personalized weight learning.
	UserID string `json:"user_id,omitempty"`

	// SelectedModel is the model that was actually served.
	SelectedModel string `json:"selected_model"`

	// CandidateModels lists all models that were eligible at decision time.
	CandidateModels []string `json:"candidate_models"`

	// SelectionMethod records which algorithm made the routing decision.
	SelectionMethod string `json:"selection_method"`

	// SelectionScore is the score assigned to the selected model.
	SelectionScore float64 `json:"selection_score"`

	// SelectionConfidence is the confidence of the selection.
	SelectionConfidence float64 `json:"selection_confidence"`

	// AllScores maps every candidate model to its score at decision time.
	AllScores map[string]float64 `json:"all_scores,omitempty"`

	// Outcome captures post-decision quality evidence.
	Outcome OfflineOutcome `json:"outcome"`

	// TransitionEvidence captures model-switch context when this turn changed models.
	TransitionEvidence *OfflineTransitionEvidence `json:"transition_evidence,omitempty"`

	// SignalValues captures the signal strengths at decision time.
	SignalValues map[string]float64 `json:"signal_values,omitempty"`

	// Timestamp is when the routing decision was made.
	Timestamp time.Time `json:"timestamp"`
}

// OfflineOutcome captures the quality and cost signals that arrive after a
// routing decision, forming the reward signal for offline training.
type OfflineOutcome struct {
	// FeedbackType is the type of feedback observed.
	// Values: "explicit", "implicit_satisfied", "implicit_need_clarification",
	// "implicit_wrong_answer", "implicit_want_different", ""
	FeedbackType string `json:"feedback_type,omitempty"`

	// FeedbackConfidence is how certain the feedback detection was (0.0-1.0).
	FeedbackConfidence float64 `json:"feedback_confidence,omitempty"`

	// Won is true when this model was preferred in a pairwise comparison.
	Won *bool `json:"won,omitempty"`

	// Tie is true when neither model was preferred.
	Tie bool `json:"tie,omitempty"`

	// OpponentModel is the model compared against (for pairwise feedback).
	OpponentModel string `json:"opponent_model,omitempty"`

	// PromptTokens is the number of prompt tokens consumed.
	PromptTokens int `json:"prompt_tokens,omitempty"`

	// CompletionTokens is the number of completion tokens generated.
	CompletionTokens int `json:"completion_tokens,omitempty"`

	// ActualCost is the dollar cost of this request.
	ActualCost float64 `json:"actual_cost,omitempty"`

	// ResponseStatus is the HTTP status code of the response.
	ResponseStatus int `json:"response_status,omitempty"`
}

// OfflineTransitionEvidence records the context when a session switches models
// mid-conversation, so offline trainers can learn handoff penalties and cache warmth
// effects from historical data.
type OfflineTransitionEvidence struct {
	// PreviousModel is the model used on the prior turn.
	PreviousModel string `json:"previous_model"`

	// CacheWarmth is the estimated KV-cache warmth at switch time (0=cold, 1=warm).
	CacheWarmth float64 `json:"cache_warmth"`

	// CacheWarmthOK indicates whether CacheWarmth was backed by reliable evidence.
	CacheWarmthOK bool `json:"cache_warmth_ok"`

	// SwitchDecision records whether the gate allowed or blocked the switch.
	SwitchDecision string `json:"switch_decision,omitempty"`

	// NetSwitchAdvantage is the computed advantage at switch time.
	NetSwitchAdvantage float64 `json:"net_switch_advantage"`

	// HandoffPenalty is the penalty applied for switching.
	HandoffPenalty float64 `json:"handoff_penalty"`
}

// OfflineDataset is a batch of records ready for offline weight training.
type OfflineDataset struct {
	// Version is the schema version for forward compatibility.
	Version int `json:"version"`

	// CreatedAt is when this dataset was assembled.
	CreatedAt time.Time `json:"created_at"`

	// WindowStart is the earliest record timestamp in the dataset.
	WindowStart time.Time `json:"window_start"`

	// WindowEnd is the latest record timestamp in the dataset.
	WindowEnd time.Time `json:"window_end"`

	// Records are the training observations, ordered by (session_id, turn_index).
	Records []OfflineDatasetRecord `json:"records"`

	// SessionCount is the number of distinct sessions in the dataset.
	SessionCount int `json:"session_count"`

	// TransitionCount is the number of mid-session model switches.
	TransitionCount int `json:"transition_count"`
}

// OfflineDatasetBuilder assembles an OfflineDataset from a replay store reader.
type OfflineDatasetBuilder interface {
	// Build constructs an offline dataset covering [windowStart, windowEnd).
	// The implementation reads from the replay store and joins transition evidence.
	Build(ctx context.Context, windowStart, windowEnd time.Time) (*OfflineDataset, error)
}
