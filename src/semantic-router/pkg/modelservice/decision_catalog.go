package modelservice

import (
	_ "embed"
	"encoding/json"
)

//go:embed decision_catalog.generated.json
var decisionCatalogJSON []byte

// CatalogDecisionCards projects checked-in canonical runtime metadata. It
// makes no readiness claim; live runtime observations remain authoritative.
func CatalogDecisionCards() []ModelCard {
	var catalog struct {
		Models []struct {
			ID            string   `json:"id"`
			Family        string   `json:"family"`
			QuestionTypes []string `json:"question_types"`
		} `json:"models"`
	}
	if err := json.Unmarshal(decisionCatalogJSON, &catalog); err != nil {
		panic("invalid generated decision catalog: " + err.Error())
	}
	cards := make([]ModelCard, 0, len(catalog.Models))
	for _, item := range catalog.Models {
		cards = append(cards, ModelCard{ID: item.ID, Repo: item.ID, Family: item.Family, Surfaces: []string{"decisions"}, QuestionTypes: item.QuestionTypes})
	}
	return cards
}
