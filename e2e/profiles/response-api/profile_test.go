package responseapi

import (
	"encoding/json"
	"os"
	"slices"
	"testing"

	utilyaml "k8s.io/apimachinery/pkg/util/yaml"
)

// The image-file E2E case requires an explicitly text-only model alongside the
// vision simulator. A name-only card is unannotated, not a text-only declaration.
func TestImageFileFixtureDeclaresModelCapabilities(t *testing.T) {
	raw, err := os.ReadFile("values.yaml")
	if err != nil {
		t.Fatal(err)
	}
	document, err := utilyaml.ToJSON(raw)
	if err != nil {
		t.Fatal(err)
	}
	var values struct {
		Config struct {
			Routing struct {
				ModelCards []struct {
					Name         string   `json:"name"`
					Capabilities []string `json:"capabilities"`
				} `json:"modelCards"`
			} `json:"routing"`
		} `json:"config"`
	}
	if err := json.Unmarshal(document, &values); err != nil {
		t.Fatal(err)
	}
	cards := make(map[string][]string)
	for _, card := range values.Config.Routing.ModelCards {
		cards[card.Name] = card.Capabilities
	}
	for model, want := range map[string][]string{
		"openai/gpt-oss-20b": {"chat"},
		"mock/vision":        {"chat", "image_input"},
	} {
		if !slices.Equal(cards[model], want) {
			t.Errorf("%s capabilities = %v, want explicit %v", model, cards[model], want)
		}
	}
}
