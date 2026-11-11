package selection

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func textRequest(text string) *llmprotocol.Request {
	return &llmprotocol.Request{Messages: []llmprotocol.Message{{
		Role:    llmprotocol.RoleUser,
		Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: text}},
	}}}
}

func outputDecision(kinds ...string) *config.Decision {
	return &config.Decision{Name: "d", Output: &config.DecisionOutput{Kind: config.OutputKinds(kinds)}}
}

func TestApplyDecisionOutputAddsTheHostedImageOperation(t *testing.T) {
	image := textRequest("draw a cat")
	if !ApplyDecisionOutput(image, outputDecision(config.OutputKindImage)) ||
		image.ImageGeneration == nil || image.ToolChoice.Mode != llmprotocol.ToolChoiceImageGeneration {
		t.Fatalf("an image decision: %+v", image)
	}
	both := textRequest("explain a cat and draw one")
	if !ApplyDecisionOutput(both, outputDecision(config.OutputKindText, config.OutputKindImage)) ||
		both.ImageGeneration == nil || both.ToolChoice.Mode != "" {
		t.Fatalf("text with an image: %+v", both)
	}
	named := textRequest("draw a cat")
	named.ToolChoice = llmprotocol.ToolChoice{Mode: llmprotocol.ToolChoiceNamed, Name: "lookup"}
	if !ApplyDecisionOutput(named, outputDecision(config.OutputKindImage)) || named.ToolChoice.Name != "lookup" {
		t.Fatalf("the request's own tool choice must stay: %+v", named)
	}
	for _, decision := range []*config.Decision{nil, {Name: "text"}, outputDecision(config.OutputKindText)} {
		if request := textRequest("hello"); ApplyDecisionOutput(request, decision) || request.ImageGeneration != nil {
			t.Fatalf("decision %+v changed a text request", decision)
		}
	}
	asked := textRequest("draw a cat")
	asked.ImageGeneration = &llmprotocol.ImageGenerationOptions{Size: "512x512"}
	if ApplyDecisionOutput(asked, outputDecision(config.OutputKindImage)) || asked.ImageGeneration.Size != "512x512" {
		t.Fatalf("the request's own operation must stay: %+v", asked)
	}
}

// The contract's capabilities join the request's: a candidate filter on
// declared capabilities sees image_generation for a text-only request.
func TestTheOutputContractJoinsTheCandidateDemand(t *testing.T) {
	request := textRequest("draw a cat")
	demand, err := EffectiveCandidateDemand(request, outputDecision(config.OutputKindImage))
	if err != nil {
		t.Fatal(err)
	}
	if !demand.ModelCapabilities.Supports(llmprotocol.CapabilityImageGeneration) || !demand.Capabilities.Supports(llmprotocol.CapabilityImageGeneration) {
		t.Fatalf("demand = %+v", demand)
	}
	if request.ImageGeneration != nil {
		t.Fatal("the candidate view changed the ingress request")
	}
	requirements := &config.CandidateRequirements{Capabilities: config.CandidateCapabilitiesDeclared}
	if err := ValidateCandidateRequirements(requirements, "chat", config.ModelParams{Capabilities: []string{"chat"}}, demand); err == nil {
		t.Fatal("a model without image_generation qualified for an image decision")
	}
	// The forced operation demands no tools capability of the model.
	if err := ValidateCandidateRequirements(requirements, "painter", config.ModelParams{Capabilities: []string{"text", "image_generation"}}, demand); err != nil {
		t.Fatalf("an image model did not qualify: %v", err)
	}
}
