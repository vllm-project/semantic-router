//go:build !windows && cgo

package apiserver

import (
	"context"
	"net/http"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
)

// FeedbackRequest represents a request to submit model selection feedback
type FeedbackRequest struct {
	// Query is the original query that was processed
	Query string `json:"query,omitempty"`

	// WinnerModel is the model that was preferred (required)
	WinnerModel string `json:"winner_model"`

	// LoserModel is the model that was not preferred (optional)
	LoserModel string `json:"loser_model,omitempty"`

	// Tie indicates if both models performed equally
	Tie bool `json:"tie,omitempty"`

	// DecisionName is the category/decision context (optional)
	DecisionName string `json:"decision_name,omitempty"`

	// Category for routing context (alias for DecisionName for RL-driven)
	Category string `json:"category,omitempty"`

	// UserID for personalized RL-driven selection (optional)
	UserID string `json:"user_id,omitempty"`

	// SessionID for multi-turn context (optional)
	SessionID string `json:"session_id,omitempty"`

	// FeedbackType for implicit feedback (satisfied, wrong_answer, etc.)
	FeedbackType string `json:"feedback_type,omitempty"`

	// Confidence for weighted feedback (0.0-1.0)
	Confidence float64 `json:"confidence,omitempty"`
}

// FeedbackResponse represents the response from feedback submission
type FeedbackResponse struct {
	Success   bool   `json:"success"`
	Message   string `json:"message"`
	Timestamp string `json:"timestamp"`
}

// handleFeedback handles POST /api/v1/feedback for submitting model selection feedback
func (s *ClassificationAPIServer) handleFeedback(w http.ResponseWriter, r *http.Request) {
	var req FeedbackRequest
	if err := s.parseJSONRequest(r, &req); err != nil {
		s.writeErrorResponse(w, http.StatusBadRequest, "INVALID_REQUEST", err.Error())
		return
	}

	// Validate required field
	if req.WinnerModel == "" {
		s.writeErrorResponse(w, http.StatusBadRequest, "MISSING_WINNER", "winner_model is required")
		return
	}

	// Resolve category/decision name
	decisionName := req.DecisionName
	if decisionName == "" && req.Category != "" {
		decisionName = req.Category
	}

	// Create feedback object with all fields
	feedback := &selection.Feedback{
		Query:        req.Query,
		WinnerModel:  req.WinnerModel,
		LoserModel:   req.LoserModel,
		Tie:          req.Tie,
		DecisionName: decisionName,
		UserID:       req.UserID,
		SessionID:    req.SessionID,
		FeedbackType: req.FeedbackType,
		Confidence:   req.Confidence,
		Timestamp:    time.Now().Unix(),
	}

	ctx := context.Background()
	selectorsUpdated := 0

	// Try to update Elo selector if available
	if eloSelector, ok := selection.GlobalRegistry.Get(selection.MethodElo); ok {
		if err := eloSelector.UpdateFeedback(ctx, feedback); err != nil {
			logging.Warnf("[FeedbackAPI] Failed to update Elo selector: %v", err)
		} else {
			selectorsUpdated++
		}
	}

	// Try to update RL-driven selector if available
	if rlSelector, ok := selection.GlobalRegistry.Get(selection.MethodRLDriven); ok {
		if err := rlSelector.UpdateFeedback(ctx, feedback); err != nil {
			logging.Warnf("[FeedbackAPI] Failed to update RL-driven selector: %v", err)
		} else {
			selectorsUpdated++
			logging.Debugf("[FeedbackAPI] RL-driven feedback updated: winner=%s, user=%s, type=%s",
				req.WinnerModel, req.UserID, req.FeedbackType)
		}
	}

	// Try to update GMTRouter selector if available (for personalized routing)
	if gmtSelector, ok := selection.GlobalRegistry.Get(selection.MethodGMTRouter); ok {
		if err := gmtSelector.UpdateFeedback(ctx, feedback); err != nil {
			logging.Warnf("[FeedbackAPI] Failed to update GMTRouter selector: %v", err)
		} else {
			selectorsUpdated++
			logging.Debugf("[FeedbackAPI] GMTRouter feedback updated: winner=%s, user=%s",
				req.WinnerModel, req.UserID)
		}
	}

	// Require at least one selector to be configured
	if selectorsUpdated == 0 {
		s.writeErrorResponse(w, http.StatusServiceUnavailable, "NO_SELECTOR_CONFIGURED",
			"No selection algorithm configured. Enable elo or rl_driven selection to use feedback API.")
		return
	}

	logging.Infof("[FeedbackAPI] Feedback recorded: winner=%s, loser=%s, tie=%v, decision=%s, user=%s, selectors=%d",
		req.WinnerModel, req.LoserModel, req.Tie, decisionName, req.UserID, selectorsUpdated)

	// Return success response
	response := FeedbackResponse{
		Success:   true,
		Message:   "Feedback recorded successfully",
		Timestamp: time.Now().UTC().Format(time.RFC3339),
	}
	s.writeJSONResponse(w, http.StatusOK, response)
}

// RatingInfo represents a single model's rating information for API response
type RatingInfo struct {
	Model  string  `json:"model"`
	Rating float64 `json:"rating"`
	Wins   int     `json:"wins"`
	Losses int     `json:"losses"`
	Ties   int     `json:"ties"`
}

// handleGetRatings handles GET /api/v1/ratings for retrieving current Elo ratings
func (s *ClassificationAPIServer) handleGetRatings(w http.ResponseWriter, r *http.Request) {
	// Get the Elo selector from the global registry
	selector, ok := selection.GlobalRegistry.Get(selection.MethodElo)
	if !ok {
		s.writeErrorResponse(w, http.StatusServiceUnavailable, "ELO_NOT_CONFIGURED",
			"Elo selection is not configured. Enable elo selection to view ratings.")
		return
	}

	// Type assert to get the EloSelector
	eloSelector, ok := selector.(*selection.EloSelector)
	if !ok {
		s.writeErrorResponse(w, http.StatusInternalServerError, "INTERNAL_ERROR",
			"Failed to get Elo selector instance")
		return
	}

	// Get optional category filter from query params
	category := r.URL.Query().Get("category")

	// Get ratings using existing GetLeaderboard method
	leaderboard := eloSelector.GetLeaderboard(category)

	// Convert to API response format
	ratings := make([]RatingInfo, 0, len(leaderboard))
	for _, r := range leaderboard {
		ratings = append(ratings, RatingInfo{
			Model:  r.Model,
			Rating: r.Rating,
			Wins:   r.Wins,
			Losses: r.Losses,
			Ties:   r.Ties,
		})
	}

	// Format response
	categoryLabel := category
	if categoryLabel == "" {
		categoryLabel = "global"
	}

	response := map[string]interface{}{
		"ratings":   ratings,
		"category":  categoryLabel,
		"count":     len(ratings),
		"timestamp": time.Now().UTC().Format(time.RFC3339),
	}

	s.writeJSONResponse(w, http.StatusOK, response)
}
