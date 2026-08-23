package config

import (
	"bytes"
	"testing"
)

func TestMarshalManagedRecipeDocumentRemovesCompiledState(t *testing.T) {
	document, err := MarshalManagedRecipeDocument(CanonicalRouting{
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
	})
	if err != nil {
		t.Fatalf("MarshalManagedRecipeDocument() error = %v", err)
	}
	for _, forbidden := range [][]byte{
		[]byte("mdl_private"), []byte("private-model"), []byte("private-judge"),
		[]byte("modelRefs"), []byte("modelCards"), []byte("dec_answer"), []byte(`"id"`),
	} {
		if bytes.Contains(document, forbidden) {
			t.Fatalf("managed Recipe document leaked physical model selection %q: %s", forbidden, document)
		}
	}
	parsed, canonical, err := ParseManagedRecipeDocument(document)
	if err != nil {
		t.Fatalf("ParseManagedRecipeDocument() error = %v", err)
	}
	if len(parsed.Decisions) != 1 || parsed.Decisions[0].ID != "" || parsed.Decisions[0].Name != "answer" {
		t.Fatalf("managed Recipe decision changed: %+v", parsed.Decisions)
	}
	if !bytes.Equal(document, canonical) {
		t.Fatalf("managed Recipe document is not canonical: %s != %s", document, canonical)
	}
}

func TestParseManagedRecipeDocumentRejectsEveryPhysicalModelSelectionSurface(t *testing.T) {
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
			if _, _, err := ParseManagedRecipeDocument([]byte(test.document)); err == nil {
				t.Fatal("ParseManagedRecipeDocument accepted physical Model selection")
			}
		})
	}
}
