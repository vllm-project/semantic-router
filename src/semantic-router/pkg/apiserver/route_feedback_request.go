//go:build !windows && cgo

package apiserver

import (
	"errors"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
)

const feedbackTimestampFormat = time.RFC3339

type feedbackValidationError struct {
	code    string
	message string
}

func normalizeAndValidateFeedbackRequest(req *FeedbackRequest) (string, *feedbackValidationError) {
	req.DecisionName = strings.TrimSpace(req.DecisionName)
	req.Category = strings.TrimSpace(req.Category)

	if req.DecisionName != "" && req.Category != "" && req.DecisionName != req.Category {
		return "", &feedbackValidationError{
			code:    "CONFLICTING_CATEGORY",
			message: "decision_name and category must match when both are provided",
		}
	}
	if req.DecisionName == "" {
		req.DecisionName = req.Category
	}

	feedback := &selection.Feedback{
		WinnerModel:    req.WinnerModel,
		LoserModel:     req.LoserModel,
		DecisionName:   req.DecisionName,
		UserID:         req.UserID,
		SessionID:      req.SessionID,
		ConversationID: req.ConversationID,
		FeedbackType:   req.FeedbackType,
		Confidence:     req.Confidence,
	}
	if err := selection.NormalizeFeedback(feedback); err != nil {
		return "", feedbackValidationFromSelectionError(err)
	}
	if feedback.WinnerModel == "" {
		return "", &feedbackValidationError{
			code:    "MISSING_WINNER",
			message: selection.ErrFeedbackWinnerRequired.Error(),
		}
	}

	req.WinnerModel = feedback.WinnerModel
	req.LoserModel = feedback.LoserModel
	req.DecisionName = feedback.DecisionName
	req.UserID = feedback.UserID
	req.SessionID = feedback.SessionID
	req.ConversationID = feedback.ConversationID
	req.FeedbackType = feedback.FeedbackType
	return feedback.DecisionName, nil
}

func feedbackTimestamp() time.Time {
	return time.Now()
}

func feedbackValidationFromSelectionError(err error) *feedbackValidationError {
	switch {
	case errors.Is(err, selection.ErrFeedbackModelRequired):
		return &feedbackValidationError{
			code:    "MISSING_WINNER",
			message: selection.ErrFeedbackWinnerRequired.Error(),
		}
	case errors.Is(err, selection.ErrFeedbackSelfComparison):
		return &feedbackValidationError{
			code:    "INVALID_MODEL_COMPARISON",
			message: selection.ErrFeedbackSelfComparison.Error(),
		}
	case errors.Is(err, selection.ErrFeedbackConfidenceRange):
		return &feedbackValidationError{
			code:    "INVALID_CONFIDENCE",
			message: selection.ErrFeedbackConfidenceRange.Error(),
		}
	default:
		return &feedbackValidationError{
			code:    "INVALID_INPUT",
			message: err.Error(),
		}
	}
}
