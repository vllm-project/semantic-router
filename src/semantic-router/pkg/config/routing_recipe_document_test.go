package config

import (
	"bytes"
	"testing"
)

func TestMarshalRoutingRecipeDocumentRemovesCompiledState(t *testing.T) {
	routing := CanonicalRouting{
		ModelCards: []RoutingModel{{Name: "private-model-card"}},
		Decisions: []Decision{{
			ID: "dec_answer", Name: "answer",
			ModelRefs: []ModelRef{{Model: "private-model"}},
			CandidateIterations: []CandidateIterationConfig{{
				Variable: "candidate", Source: "models",
				Models: []ModelRef{{Model: "private-model"}},
			}},
			Algorithm: &AlgorithmConfig{
				Type:   "prompt",
				Prompt: &PromptSelectionConfig{Model: "private-judge", Instructions: "choose"},
			},
		}},
	}
	document, err := MarshalRoutingRecipeDocument(routing)
	if err != nil {
		t.Fatalf("MarshalRoutingRecipeDocument() error = %v", err)
	}
	for _, forbidden := range [][]byte{
		[]byte("mdl_private"), []byte("private-model"), []byte("private-judge"),
		[]byte("modelRefs"), []byte("modelCards"), []byte("dec_answer"), []byte(`"id"`),
	} {
		if bytes.Contains(document, forbidden) {
			t.Fatalf("routing Recipe document leaked physical model selection %q: %s", forbidden, document)
		}
	}
	parsed, canonical, err := ParseRoutingRecipeDocument(document)
	if err != nil {
		t.Fatalf("ParseRoutingRecipeDocument() error = %v", err)
	}
	if len(parsed.Decisions) != 1 || parsed.Decisions[0].ID != "" || parsed.Decisions[0].Name != "answer" {
		t.Fatalf("routing Recipe decision changed: %+v", parsed.Decisions)
	}
	if !bytes.Equal(document, canonical) {
		t.Fatalf("routing Recipe document is not canonical: %s != %s", document, canonical)
	}
	if routing.Decisions[0].ID != "dec_answer" || len(routing.Decisions[0].ModelRefs) != 1 ||
		routing.Decisions[0].Algorithm == nil || routing.Decisions[0].Algorithm.Prompt == nil ||
		routing.Decisions[0].Algorithm.Prompt.Model != "private-judge" {
		t.Fatalf("marshalling mutated the source routing document: %+v", routing.Decisions[0])
	}
}

func TestCanonicalRoutingFromRecipeDocumentUsesExplicitModelFreeProjection(t *testing.T) {
	document := RoutingRecipeDocument{
		Signals: CanonicalSignals{
			Keywords: []KeywordRule{{Name: "math", Keywords: []string{"calculate"}}},
		},
		Decisions: []Decision{{Name: "answer", Priority: 10}},
		Strategy:  RoutingStrategyPriority,
	}

	routing := CanonicalRoutingFromRecipeDocument(document)
	if len(routing.ModelCards) != 0 {
		t.Fatalf("Recipe projection introduced Model cards: %+v", routing.ModelCards)
	}
	if len(routing.Signals.Keywords) != 1 || routing.Signals.Keywords[0].Name != "math" {
		t.Fatalf("Recipe signals changed during projection: %+v", routing.Signals)
	}
	if len(routing.Decisions) != 1 || routing.Decisions[0].Name != "answer" {
		t.Fatalf("Recipe decisions changed during projection: %+v", routing.Decisions)
	}
	if routing.Strategy != RoutingStrategyPriority {
		t.Fatalf("Recipe strategy = %q, want %q", routing.Strategy, RoutingStrategyPriority)
	}

	routing.Decisions[0].Name = "mutated"
	if document.Decisions[0].Name != "answer" {
		t.Fatal("Recipe projection shares mutable Decision state with its source")
	}
}

func TestNormalizeEntrypointModelNamesPreservesPrimaryOrder(t *testing.T) {
	got := normalizeEntrypointModelNames([]string{" vllm-sr/blend ", "blend", "vllm-sr/blend", " "})
	want := []string{"vllm-sr/blend", "blend"}
	if len(got) != len(want) || got[0] != want[0] || got[1] != want[1] {
		t.Fatalf("normalizeEntrypointModelNames() = %v, want %v", got, want)
	}
}

func TestParseRoutingRecipeDocumentRejectsEveryPhysicalModelSelectionSurface(t *testing.T) {
	tests := []struct {
		name     string
		document string
	}{
		{name: "decision candidates", document: `{"decisions":[{"name":"answer","rules":{},"modelRefs":[{"model":"private"}]}]}`},
		{name: "candidate iteration", document: `{"decisions":[{"name":"answer","rules":{},"candidateIterations":[{"variable":"candidate","source":"models","models":[{"model":"private"}]}]}]}`},
		{name: "fusion judge", document: `{"decisions":[{"name":"answer","rules":{},"algorithm":{"type":"fusion","fusion":{"model":"private"}}}]}`},
		{name: "workflow role", document: `{"decisions":[{"name":"answer","rules":{},"algorithm":{"type":"workflows","workflows":{"roles":[{"name":"worker","models":["private"]}]}}}]}`},
		{name: "remom synthesis", document: `{"decisions":[{"name":"answer","rules":{},"algorithm":{"type":"remom","remom":{"breadth_schedule":[1],"synthesis_model":"private"}}}]}`},
		{name: "prompt selector", document: `{"decisions":[{"name":"answer","rules":{},"algorithm":{"type":"prompt","prompt":{"model":"private","instructions":"select"}}}]}`},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if _, _, err := ParseRoutingRecipeDocument([]byte(test.document)); err == nil {
				t.Fatal("ParseRoutingRecipeDocument accepted physical Model selection")
			}
		})
	}
}
