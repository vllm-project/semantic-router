package modelservice

import (
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/utils/decisionmedia"
)

// PrepareDecisionMedia applies deployment policies using observed cards only.
// The lookup must not discover models or wait for readiness on the request path.
func PrepareDecisionMedia(deployments map[string]config.ModelDeployment, sources []*decisionmedia.Source, currentCard func(string) (ModelCard, bool)) map[string]DecisionMedia {
	if len(sources) == 0 || currentCard == nil {
		return nil
	}
	result := make(map[string]DecisionMedia)
	for name, deployment := range deployments {
		if deployment.Input.Media == nil {
			continue
		}
		card, ok := currentCard(name)
		if !ok || ValidateDecisionMedia(deployment, card) != nil {
			continue
		}
		policy := decisionmedia.Policy{
			Images: decisionMediaLimit(deployment.Input.Media.Images, 4, card.ImageMaxBytes),
			Videos: decisionMediaLimit(deployment.Input.Media.Videos, 1, card.VideoMaxBytes),
		}
		selected := decisionmedia.Select(sources, policy, RecordMediaSkip)
		result[name] = DecisionMedia{Images: selected.Images, Videos: selected.Videos}
	}
	return result
}

func decisionMediaLimit(limit *config.DecisionMediaLimit, count, bytes int) *decisionmedia.Limit {
	if limit == nil {
		return nil
	}
	if limit.MaxCount > 0 {
		count = limit.MaxCount
	}
	if limit.MaxBytes > 0 {
		bytes = limit.MaxBytes
	}
	return &decisionmedia.Limit{Count: count, Bytes: bytes}
}
