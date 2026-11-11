package modelservice

import "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"

// CatalogDecisionCards projects checked-in canonical runtime metadata. It
// makes no readiness claim; live runtime observations remain authoritative.
func CatalogDecisionCards() []ModelCard {
	models := config.DecisionCatalog()
	cards := make([]ModelCard, 0, len(models))
	for _, model := range models {
		cards = append(cards, ModelCard{
			ID: model.ID, Repo: model.ID, Family: model.Family, Surfaces: []string{"decisions"},
			QuestionTypes: model.QuestionTypes, Modalities: model.Modalities,
		})
	}
	return cards
}
