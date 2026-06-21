package extproc

import (
	"math"
	"sort"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
)

func (rt *routerLearningRuntime) updateEloFeedback(feedback *selection.Feedback) bool {
	cfg, ok := rt.eloFeedbackConfig()
	if !ok {
		return false
	}
	stateKey, ok := learningStateKeyFromParts(
		cfg.EffectiveScope(),
		feedback.DecisionName,
		feedback.SessionID,
		feedback.ConversationID,
	)
	if !ok {
		return false
	}
	rt.updateEloRatings(stateKey, cfg, feedback)
	return true
}

func (rt *routerLearningRuntime) eloFeedbackConfig() (config.EloLearningConfig, bool) {
	if rt == nil || rt.config == nil || !rt.config.RouterLearning.Enabled {
		return config.EloLearningConfig{}, false
	}
	cfg := rt.config.RouterLearning.Adaptations.Elo
	return cfg, cfg.Enabled
}

func (rt *routerLearningRuntime) updateEloRatings(
	stateKey string,
	cfg config.EloLearningConfig,
	feedback *selection.Feedback,
) {
	if rt == nil || strings.TrimSpace(stateKey) == "" || feedback == nil {
		return
	}
	rt.mu.Lock()
	defer rt.mu.Unlock()

	ratings := rt.eloRatings(stateKey)
	switch {
	case feedback.WinnerModel != "" && feedback.LoserModel != "":
		rt.applyEloPairwiseFeedbackLocked(ratings, cfg, feedback)
	case feedback.WinnerModel != "":
		rt.applyEloWinnerOnlyFeedbackLocked(ratings, cfg, feedback.WinnerModel)
	case feedback.LoserModel != "":
		rt.applyEloLoserOnlyFeedbackLocked(ratings, cfg, feedback.LoserModel)
	}
}

func (rt *routerLearningRuntime) applyEloPairwiseFeedbackLocked(
	ratings map[string]*routerLearningEloRating,
	cfg config.EloLearningConfig,
	feedback *selection.Feedback,
) {
	winner := eloRatingLocked(ratings, cfg, feedback.WinnerModel)
	loser := eloRatingLocked(ratings, cfg, feedback.LoserModel)

	expectedWinner := 1.0 / (1.0 + math.Pow(10, (loser.Rating-winner.Rating)/400.0))
	expectedLoser := 1.0 - expectedWinner
	actualWinner, actualLoser := 1.0, 0.0
	if feedback.Tie {
		actualWinner = 0.5
		actualLoser = 0.5
	}
	k := eloKFactor(cfg)
	winner.Rating += k * (actualWinner - expectedWinner)
	loser.Rating += k * (actualLoser - expectedLoser)
	winner.Comparisons++
	loser.Comparisons++
	if feedback.Tie {
		winner.Ties++
		loser.Ties++
	} else {
		winner.Wins++
		loser.Losses++
	}
	now := time.Now()
	winner.LastUpdated = now
	loser.LastUpdated = now
}

func (rt *routerLearningRuntime) applyEloWinnerOnlyFeedbackLocked(
	ratings map[string]*routerLearningEloRating,
	cfg config.EloLearningConfig,
	model string,
) {
	rating := eloRatingLocked(ratings, cfg, model)
	rating.Rating += eloKFactor(cfg) * 0.1
	rating.Comparisons++
	rating.Wins++
	rating.LastUpdated = time.Now()
}

func (rt *routerLearningRuntime) applyEloLoserOnlyFeedbackLocked(
	ratings map[string]*routerLearningEloRating,
	cfg config.EloLearningConfig,
	model string,
) {
	rating := eloRatingLocked(ratings, cfg, model)
	rating.Rating -= eloKFactor(cfg) * 0.1
	rating.Comparisons++
	rating.Losses++
	rating.LastUpdated = time.Now()
}

func (rt *routerLearningRuntime) eloRatings(stateKey string) map[string]*routerLearningEloRating {
	if rt.elo.ratings == nil {
		rt.elo.ratings = map[string]map[string]*routerLearningEloRating{}
	}
	if rt.elo.ratings[stateKey] == nil {
		rt.elo.ratings[stateKey] = map[string]*routerLearningEloRating{}
	}
	return rt.elo.ratings[stateKey]
}

func eloRatingLocked(
	ratings map[string]*routerLearningEloRating,
	cfg config.EloLearningConfig,
	model string,
) *routerLearningEloRating {
	model = strings.TrimSpace(model)
	if ratings[model] == nil {
		ratings[model] = &routerLearningEloRating{
			Model:  model,
			Rating: eloInitialRating(cfg),
		}
	}
	return ratings[model]
}

func (rt *routerLearningRuntime) EloLeaderboard(category string) []selection.ModelRating {
	if !rt.EloLearningEnabled() {
		return nil
	}
	stateKey, ok := learningStateKeyFromParts(config.RouterLearningScopeDecision, category, "", "")
	if !ok {
		return nil
	}
	if rt != nil {
		rt.mu.Lock()
		defer rt.mu.Unlock()
	}
	if rt == nil || rt.elo.ratings == nil || rt.elo.ratings[stateKey] == nil {
		return nil
	}
	leaderboard := make([]selection.ModelRating, 0, len(rt.elo.ratings[stateKey]))
	for _, rating := range rt.elo.ratings[stateKey] {
		if rating == nil {
			continue
		}
		leaderboard = append(leaderboard, selection.ModelRating{
			Model:       rating.Model,
			Rating:      rating.Rating,
			Comparisons: rating.Comparisons,
			Wins:        rating.Wins,
			Losses:      rating.Losses,
			Ties:        rating.Ties,
		})
	}
	sort.SliceStable(leaderboard, func(i, j int) bool {
		if leaderboard[i].Rating == leaderboard[j].Rating {
			return leaderboard[i].Model < leaderboard[j].Model
		}
		return leaderboard[i].Rating > leaderboard[j].Rating
	})
	return leaderboard
}

func (rt *routerLearningRuntime) EloLearningEnabled() bool {
	if rt == nil || rt.config == nil || !rt.config.RouterLearning.Enabled {
		return false
	}
	return rt.config.RouterLearning.Adaptations.Elo.Enabled
}
