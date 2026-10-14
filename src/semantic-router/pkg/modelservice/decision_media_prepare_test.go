package modelservice

import (
	"encoding/base64"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/utils/decisionmedia"
)

func TestPrepareDecisionMediaUsesEachDeploymentPolicyAndCurrentCard(t *testing.T) {
	image := func(text string) string {
		return "data:image/png;base64," + base64.StdEncoding.EncodeToString([]byte("\x89PNG\r\n\x1a\n"+text))
	}
	old, recent := image("old"), image("recent")
	sources := []*decisionmedia.Source{{Kind: "image", URL: old}, {Kind: "image", URL: recent}}
	enabled := config.ModelDeployment{Provider: config.ModelRuntimeProvider, Input: config.ModelInputBudget{Media: &config.DecisionMediaConfig{Images: &config.DecisionMediaLimit{MaxCount: 1}}}}
	small := enabled
	small.Input.Media = &config.DecisionMediaConfig{Images: &config.DecisionMediaLimit{MaxBytes: 8}}
	deployments := map[string]config.ModelDeployment{"vision": enabled, "limited": small, "offline": enabled, "text": enabled, "off": {Provider: config.ModelRuntimeProvider}}
	ready := false
	lookup := func(name string) (ModelCard, bool) {
		require.NotEqual(t, "off", name, "default-off deployment must not consult a card")
		card := ModelCard{Surfaces: []string{"decisions"}, Modalities: []string{"text", "image"}, ImageMaxBytes: 8000000}
		if name == "text" {
			card.Modalities = []string{"text"}
		}
		return card, name != "offline" || ready
	}
	media := PrepareDecisionMedia(deployments, sources, lookup)
	require.Equal(t, []string{recent}, media["vision"].Images)
	require.Empty(t, media["limited"].Images)
	for _, name := range []string{"offline", "text", "off"} {
		require.NotContains(t, media, name)
	}
	ready = true
	media = PrepareDecisionMedia(deployments, sources, lookup)
	require.Equal(t, []string{recent}, media["offline"].Images)
}
