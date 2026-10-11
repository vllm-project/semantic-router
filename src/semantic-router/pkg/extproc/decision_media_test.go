package extproc

import (
	"encoding/base64"
	"fmt"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
)

// This exercises ingress -> neutral extraction -> deployment policy -> selector.
// The fake records inputs; it makes no claim about real-model visual reasoning.
func TestDecisionMediaReachesSelectorFromChat(t *testing.T) {
	image := "data:image/png;base64," + base64.StdEncoding.EncodeToString([]byte("\x89PNG\r\n\x1a\nfixture"))
	for _, enabled := range []bool{false, true} {
		cfg := decisionSelectorConfig(t, "")
		if enabled {
			d := cfg.ModelDeployments["primary"]
			d.Input.Media = &config.DecisionMediaConfig{Images: &config.DecisionMediaLimit{}}
			cfg.ModelDeployments["primary"] = d
		}
		decider := &recordingDecider{card: &modelservice.ModelCard{Surfaces: []string{"decisions"}, Modalities: []string{"text", "image"}, ImageMaxBytes: 8000000}, answer: modelservice.Answer{Type: "choice", Choice: "deep", Probabilities: map[string]float64{"fast": 0.1, "deep": 0.9}}}
		router := preparedSelectorRouter(t, cfg, decider)
		body := []byte(fmt.Sprintf(`{"model":"auto","messages":[{"role":"user","content":[{"type":"text","text":"Explain this"},{"type":"image_url","image_url":{"url":%q}}]}]}`, image))
		neutral, _, _, err := protocolcodec.NewBuiltinEngine().DecodeRequestForMutation(llmprotocol.OpenAIChatV1, body)
		if err != nil {
			t.Fatal(err)
		}
		snapshot := extractSemanticRequestSignals(&neutral)
		ctx := &RequestContext{TraceContext: t.Context(), decisionMedia: router.prepareRequestDecisionMedia(snapshot.DecisionMedia)}
		decision := cfg.GetDecisionByName("choose")
		result, _, err := router.selectWithSelector(
			&selection.SelectionContext{Query: snapshot.UserContent, CandidateModels: decision.ModelRefs},
			selection.MethodDecision,
			router.newDecisionModelSelector(*decision.Algorithm.Decision),
			&decision.ModelRefs[0], ctx,
		)
		if err != nil {
			t.Fatal(err)
		}
		if result == nil || result.Model != "deep" || len(decider.requests) != 1 {
			t.Fatal("selector call failed")
		}
		got := decider.requests[0].Images
		if enabled && (len(got) != 1 || got[0] != image) {
			t.Fatal("opted-in selector lost image")
		}
		if !enabled && len(got) != 0 {
			t.Fatal("default-off selector received image")
		}
		if neutral.Messages[0].Content[1].Data != image[len("data:image/png;base64,"):] {
			t.Fatal("source request changed")
		}
	}
}

func TestDecisionMediaStartupRejectsTextCard(t *testing.T) {
	cfg := decisionSelectorConfig(t, "")
	d := cfg.ModelDeployments["primary"]
	d.Input.Media = &config.DecisionMediaConfig{Images: &config.DecisionMediaLimit{}}
	cfg.ModelDeployments["primary"] = d
	if _, err := prepareDecisionSelectionCards(cfg, &recordingDecider{}); err == nil {
		t.Fatal("unsupported media admitted")
	}
}
