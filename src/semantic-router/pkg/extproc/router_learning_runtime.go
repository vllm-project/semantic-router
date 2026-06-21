package extproc

import (
	"context"
	"sync"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
)

type routerLearningRuntime struct {
	mu              sync.Mutex
	config          *config.RouterConfig
	bandit          routerLearningBanditState
	elo             routerLearningEloState
	personalization routerLearningPersonalizationState
}

func newRouterLearningRuntime(cfg *config.RouterConfig) *routerLearningRuntime {
	return &routerLearningRuntime{
		config: cfg,
		bandit: routerLearningBanditState{
			arms: map[string]map[string]*routerLearningBanditArmState{},
		},
		elo: routerLearningEloState{
			ratings: map[string]map[string]*routerLearningEloRating{},
		},
		personalization: routerLearningPersonalizationState{
			preferences: map[string]map[string]*routerLearningPersonalizationModelState{},
		},
	}
}

func (r *OpenAIRouter) routerLearningRuntimeState() *routerLearningRuntime {
	if r == nil {
		return nil
	}
	r.routerLearningMu.Lock()
	defer r.routerLearningMu.Unlock()
	if r.routerLearningRuntime == nil {
		r.routerLearningRuntime = newRouterLearningRuntime(r.Config)
	}
	return r.routerLearningRuntime
}

func (rt *routerLearningRuntime) UpdateFeedback(_ context.Context, feedback *selection.Feedback) int {
	if rt == nil || feedback == nil {
		return 0
	}
	if err := selection.NormalizeFeedback(feedback); err != nil {
		return 0
	}
	updated := 0
	if rt.updateBanditFeedback(feedback) {
		updated++
	}
	if rt.updateEloFeedback(feedback) {
		updated++
	}
	if rt.updatePersonalizationFeedback(feedback) {
		updated++
	}
	return updated
}
