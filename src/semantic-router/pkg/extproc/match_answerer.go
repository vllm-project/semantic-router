package extproc

import (
	"context"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/match"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
)

// routerMatchAnswerer answers the judges of match blocks: a systemone
// deployment through the generation's System One client, a model_runtime one
// through its model lease, with the card the generation prepared for it.
type routerMatchAnswerer struct{ r *OpenAIRouter }

func (a routerMatchAnswerer) Decide(ctx context.Context, deployment string, request modelservice.Request) (modelservice.Response, error) {
	if a.r.Config.ModelDeployments[deployment].IsSystemOne() {
		if a.r.systemOneDecider == nil {
			return modelservice.Response{}, modelservice.ErrUnavailable
		}
		return a.r.systemOneDecider.Decide(ctx, deployment, request)
	}
	decider := a.r.decisionDecider
	if decider == nil {
		decider = modelservice.Default()
	}
	return decider.Decide(ctx, deployment, request)
}

func (a routerMatchAnswerer) Card(deployment string) (modelservice.ModelCard, bool) {
	if a.r.Config.ModelDeployments[deployment].IsSystemOne() {
		if a.r.systemOneDecider == nil {
			return modelservice.ModelCard{}, false
		}
		return a.r.systemOneDecider.Card(deployment)
	}
	card, ok := a.r.decisionCards[deployment]
	return card, ok
}

// matcher runs the match blocks of the router's configuration.
func (r *OpenAIRouter) matcher() *match.Matcher {
	return match.New(r.Config, routerMatchAnswerer{r})
}

// matchJudges are the judges of the match blocks the reachable decisions
// run, for a generation to prepare the deployments they ask.
func matchJudges(cfg *config.RouterConfig) []config.MatchJudge {
	var judges []config.MatchJudge
	for _, recipe := range cfg.ReachableRoutingRecipes() {
		for index := range recipe.Profile.Decisions {
			judges = append(judges, recipe.Profile.Decisions[index].MatchJudges()...)
		}
	}
	return judges
}
