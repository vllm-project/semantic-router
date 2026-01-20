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

	// Get the Elo selector from the global registry
	selector, ok := selection.GlobalRegistry.Get(selection.MethodElo)
	if !ok {
		s.writeErrorResponse(w, http.StatusServiceUnavailable, "ELO_NOT_CONFIGURED",
			"Elo selection is not configured. Enable elo selection to use feedback API.")
		return
	}

	// Create feedback object
	feedback := &selection.Feedback{
		Query:        req.Query,
		WinnerModel:  req.WinnerModel,
		LoserModel:   req.LoserModel,
		Tie:          req.Tie,
		DecisionName: req.DecisionName,
		Timestamp:    time.Now().Unix(),
	}

	// Submit feedback to the selector
	ctx := context.Background()
	if err := selector.UpdateFeedback(ctx, feedback); err != nil {
		logging.Errorf("[FeedbackAPI] Failed to update feedback: %v", err)
		s.writeErrorResponse(w, http.StatusInternalServerError, "FEEDBACK_FAILED", err.Error())
		return
	}

	logging.Infof("[FeedbackAPI] Feedback recorded: winner=%s, loser=%s, tie=%v, decision=%s",
		req.WinnerModel, req.LoserModel, req.Tie, req.DecisionName)

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
