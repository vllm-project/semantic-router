package config

import (
	"strings"
	"testing"

	"gopkg.in/yaml.v3"
)

// outputContractDocument is a configuration whose image decision declares
// output and references model, beside a text decision on the chat model.
func outputContractDocument(output, model string) string {
	return `version: v0.5
providers:
  defaults: {model: chat}
  models:
    - name: chat
      backend_refs: [{name: chat, endpoint: "127.0.0.1:8000", protocol: http, weight: 1}]
    - name: painter
      api_format: images
      backend_refs: [{name: images, endpoint: "127.0.0.1:8001", protocol: http, weight: 1}]
      capabilities: [image_generation]
      modalities: {input: [text], output: [image]}
    - name: omni
      api_format: responses
      backend_refs: [{name: omni, endpoint: "127.0.0.1:8002", protocol: http, weight: 1}]
      capabilities: [chat, image_generation, image_input, audio_output, video_output, video_input]
      modalities: {input: [text, image, video], output: [text, image, audio, video]}
    - name: chat-painter
      backend_refs: [{name: chat-painter, endpoint: "127.0.0.1:8003", protocol: http, weight: 1}]
      capabilities: [image_generation]
recipes:
  - name: default
    decisions:
      - name: picture
        priority: 100
        rules: {operator: AND, conditions: []}
        ` + output + `
        modelRefs: [{model: ` + model + `}]
      - name: everything
        rules: {}
        modelRefs: [{model: chat}]
`
}

func loadOutputContract(t *testing.T, output, model string) error {
	t.Helper()
	_, err := ParseYAMLBytesWithoutEnvExpansion([]byte(outputContractDocument(output, model)))
	return err
}

func TestOutputContractsThisReleaseExecutesLoad(t *testing.T) {
	for _, tc := range []struct{ output, model string }{
		{"", "chat"},
		{"output: {kind: text}", "chat"},
		{"output: {kind: image, action: generate}", "painter"},
		{"output: {kind: image}", "omni"},
		{"output: {kind: [text, image], action: generate}", "omni"},
	} {
		if err := loadOutputContract(t, tc.output, tc.model); err != nil {
			t.Errorf("%q on %s: %v", tc.output, tc.model, err)
		}
	}
}

func TestEveryModelReferenceDeclaresTheCapabilitiesItsContractNeeds(t *testing.T) {
	for _, tc := range []struct{ output, model, want string }{
		{"output: {kind: image, action: generate}", "chat", `model "chat" does not declare image_generation in capabilities, which kind image needs`},
		{"output: {kind: [text, image], action: generate}", "painter", `model "painter" does not declare text in modalities.output, which kind [text, image] needs`},
		{"output: {kind: [text, image]}", "chat", `model "chat" does not declare image_generation in capabilities, which kind [text, image] needs`},
		{"", "painter", `model "painter" does not declare text in modalities.output, which kind text (the default) needs`},
	} {
		err := loadOutputContract(t, tc.output, tc.model)
		want := "recipes[default].decisions[picture].output: output_capability_missing: " + tc.want
		if err == nil || !strings.Contains(err.Error(), want) {
			t.Errorf("%q on %s: got %v, want %s", tc.output, tc.model, err, want)
		}
	}
}

// Kinds and actions without a codec in this release fail whatever the model
// declares, naming the codec they need.
func TestLaterPhaseOutputsAreNotExecutable(t *testing.T) {
	for _, tc := range []struct{ output, codec string }{
		{"output: {kind: image, action: edit}", "the image edit codec"},
		{"output: {kind: [text, image], action: edit}", "the image edit codec"},
		{"output: {kind: audio}", "the audio output codec"},
		{"output: {kind: [text, audio]}", "the audio output codec"},
		{"output: {kind: video}", "the video codec"},
		{"output: {kind: video, action: edit}", "the video codec"},
	} {
		err := loadOutputContract(t, tc.output, "omni")
		if err == nil || !strings.Contains(err.Error(), "recipes[default].decisions[picture].output: output_not_executable: ") ||
			!strings.Contains(err.Error(), "needs "+tc.codec+", which this release does not have") {
			t.Errorf("%q: got %v, want output_not_executable naming %s", tc.output, err, tc.codec)
		}
	}
}

// A model whose API cannot carry the contract's output never serves it.
func TestOutputContractNeedsAnAPIThatCarriesIt(t *testing.T) {
	for _, tc := range []struct{ output, model, want string }{
		{"output: {kind: image}", "chat-painter", `model "chat-painter" answers on the chat_completions API, which cannot carry a generated image; reach it with api_format: images or responses`},
	} {
		err := loadOutputContract(t, tc.output, tc.model)
		if err == nil || !strings.Contains(err.Error(), "output_not_executable: "+tc.want) {
			t.Errorf("%q on %s: got %v, want %s", tc.output, tc.model, err, tc.want)
		}
	}
	text := strings.Replace(outputContractDocument("output: {kind: [text, image]}", "omni"), "api_format: responses", "api_format: images", 1)
	if _, err := ParseYAMLBytesWithoutEnvExpansion([]byte(text)); err == nil ||
		!strings.Contains(err.Error(), `model "omni" answers on the images API, which carries no text; reach it with api_format: responses`) {
		t.Errorf("text with an image on the images API: %v", err)
	}
}

func TestAnExecutorAnswersInTextOnly(t *testing.T) {
	decision := Decision{
		Name:      "picture",
		Output:    &DecisionOutput{Kind: OutputKinds{OutputKindImage}},
		Algorithm: &AlgorithmConfig{Type: DecisionAlgorithmEscalate},
	}
	err := validateDecisionOutput(&RouterConfig{}, decision)
	if err == nil || err.Error() != "recipes[default].decisions[picture].output: output_not_executable: algorithm escalate is an executor, whose model calls answer in text only; select one model for kind image" {
		t.Fatalf("an escalating image decision: %v", err)
	}
	decision.Output = &DecisionOutput{Kind: OutputKinds{OutputKindText}}
	if err := validateDecisionOutput(&RouterConfig{}, decision); err != nil {
		t.Fatalf("an escalating text decision: %v", err)
	}
}

func TestOutputContractValues(t *testing.T) {
	for _, tc := range []struct{ output, want string }{
		{"output: {kind: picture}", `decisions[picture].output: kind "picture" is not one of text, image, audio, video`},
		{"output: {kind: [image, image]}", "decisions[picture].output: kind lists image twice"},
		{"output: {kind: []}", "decisions[picture].output: kind lists no output"},
		{"output: {kind: text, action: edit}", "action edit edits an image or a video the request provides, and kind text has neither"},
		{"output: {kind: image, action: draw}", `action "draw" is not one of generate, edit`},
	} {
		if err := loadOutputContract(t, tc.output, "omni"); err == nil || !strings.Contains(err.Error(), tc.want) {
			t.Errorf("%q: got %v, want %s", tc.output, err, tc.want)
		}
	}
}

// One kind is written as a scalar and several as a list, both ways.
func TestOutputKindsReadAndWriteAScalarOrAList(t *testing.T) {
	for _, document := range []string{"kind: image\naction: generate\n", "kind:\n    - text\n    - image\n"} {
		var output DecisionOutput
		if err := yaml.Unmarshal([]byte(document), &output); err != nil {
			t.Fatal(err)
		}
		written, err := yaml.Marshal(output)
		if err != nil {
			t.Fatal(err)
		}
		if string(written) != document {
			t.Errorf("wrote %q, want %q", written, document)
		}
	}
	var unset *DecisionOutput
	if !unset.Includes(OutputKindText) || unset.GeneratesImages() || unset.EffectiveAction() != OutputActionGenerate {
		t.Fatal("a decision without a contract answers in text")
	}
	both := &DecisionOutput{Kind: OutputKinds{OutputKindText, OutputKindImage}}
	if !both.GeneratesImages() || both.ImageOnly() || both.String() != "[text, image]" {
		t.Fatalf("text with an image: %+v", both)
	}
}
