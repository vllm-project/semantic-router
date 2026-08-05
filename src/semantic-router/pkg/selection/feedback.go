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
	"math"
	"strings"
)

var (
	ErrFeedbackRequired        = errors.New("feedback is required")
	ErrFeedbackModelRequired   = errors.New("either winner_model or loser_model is required")
	ErrFeedbackWinnerRequired  = errors.New("winner_model is required")
	ErrFeedbackSelfComparison  = errors.New("winner_model and loser_model must be different models")
	ErrFeedbackConfidenceRange = errors.New("confidence must be between 0.0 and 1.0")
)

// NormalizeFeedback canonicalizes fields that identify selector state and then
// validates common feedback contracts before a selector mutates its model state.
func NormalizeFeedback(feedback *Feedback) error {
	if feedback == nil {
		return ErrFeedbackRequired
	}

	feedback.WinnerModel = strings.TrimSpace(feedback.WinnerModel)
	feedback.LoserModel = strings.TrimSpace(feedback.LoserModel)
	feedback.DecisionName = strings.TrimSpace(feedback.DecisionName)
	feedback.UserID = strings.TrimSpace(feedback.UserID)
	feedback.SessionID = strings.TrimSpace(feedback.SessionID)
	feedback.ConversationID = strings.TrimSpace(feedback.ConversationID)
	feedback.FeedbackType = strings.TrimSpace(feedback.FeedbackType)

	return ValidateFeedback(feedback)
}

// ValidateFeedback checks selector-wide feedback contracts without mutating the
// supplied feedback object.
func ValidateFeedback(feedback *Feedback) error {
	if feedback == nil {
		return ErrFeedbackRequired
	}

	winnerModel := strings.TrimSpace(feedback.WinnerModel)
	loserModel := strings.TrimSpace(feedback.LoserModel)
	if winnerModel == "" && loserModel == "" {
		return ErrFeedbackModelRequired
	}

	if winnerModel != "" && loserModel != "" && winnerModel == loserModel {
		return ErrFeedbackSelfComparison
	}

	if math.IsNaN(feedback.Confidence) || math.IsInf(feedback.Confidence, 0) ||
		feedback.Confidence < 0 || feedback.Confidence > 1 {
		return ErrFeedbackConfidenceRange
	}

	return nil
}

func normalizeWinnerFeedback(feedback *Feedback) error {
	if err := NormalizeFeedback(feedback); err != nil {
		return err
	}
	if feedback.WinnerModel == "" {
		return ErrFeedbackWinnerRequired
	}
	return nil
}
