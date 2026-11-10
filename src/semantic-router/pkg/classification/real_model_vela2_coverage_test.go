package classification

import (
	"encoding/json"
	"fmt"
	"reflect"
	"testing"
)

// The published corpus's non-truncating tasks explicitly require full input.
// A digest-only exclusion must never let absent request flags or worker proof
// pass the real-model contract. Numerical/model-prompt goldens remain separate.
func checkVela2InputCoverage(questions map[string]json.RawMessage, response map[string]interface{}) error {
	answers, _ := response["answers"].(map[string]interface{})
	sets, _ := response["sets"].(map[string]interface{})
	proof := func(id string, value interface{}) error {
		answer, ok := value.(map[string]interface{})
		if !ok || answer["input_coverage"] != "complete" || answer["error"] != nil {
			return fmt.Errorf("%s: strict published question has no successful complete-input proof", id)
		}
		return nil
	}
	for id, raw := range questions {
		var question struct {
			Type             string `json:"type"`
			Overflow         string `json:"overflow"`
			RequireFullInput bool   `json:"require_full_input"`
		}
		if err := json.Unmarshal(raw, &question); err != nil {
			return err
		}
		strict := question.Overflow != "truncate"
		if question.RequireFullInput != strict {
			return fmt.Errorf("%s: require_full_input=%t, want %t for overflow %q", id, question.RequireFullInput, strict, question.Overflow)
		}
		if !strict {
			continue
		}
		if question.Type != "set" {
			if err := proof(id, answers[id]); err != nil {
				return err
			}
			continue
		}
		if err := proof(id, sets[id]); err != nil {
			return err
		}
		set := sets[id].(map[string]interface{})
		labels, _ := set["probabilities"].(map[string]interface{})
		if len(labels) == 0 {
			return fmt.Errorf("%s: strict Set has no label answers", id)
		}
		for label := range labels {
			if err := proof(id+"."+label, answers[id+"."+label]); err != nil {
				return err
			}
		}
	}
	return nil
}

func TestVela2CoverageMetadataIsNotModelInput(t *testing.T) {
	original := map[string]json.RawMessage{"q": json.RawMessage(`{"type":"set","criteria":{"z":"last","a":"first"},"instructions":"Classify"}`)}
	strict := map[string]json.RawMessage{"q": json.RawMessage(`{"type":"set","require_full_input":true,"criteria":{"z":"last","a":"first"},"instructions":"Classify"}`)}
	if questionsDigest(original) != questionsDigest(strict) {
		t.Fatal("admission metadata changed model-input digest")
	}
	for _, altered := range []string{
		`{"type":"set","criteria":{"a":"first","z":"last"},"instructions":"Classify"}`,
		`{"type":"set","criteria":{"z":"last","a":"first"},"instructions":"Different"}`,
	} {
		if questionsDigest(original) == questionsDigest(map[string]json.RawMessage{"q": json.RawMessage(altered)}) {
			t.Fatal("model-facing change escaped digest")
		}
	}
}

func TestVela2PublishedCoverageProofIsMandatory(t *testing.T) {
	questions := map[string]json.RawMessage{"q": json.RawMessage(`{"type":"noul","require_full_input":true}`)}
	complete := map[string]interface{}{"type": "noul", "noul": 0.4, "input_coverage": "complete"}
	response := map[string]interface{}{"answers": map[string]interface{}{"q": complete}}
	if err := checkVela2InputCoverage(questions, response); err != nil {
		t.Fatal(err)
	}
	if err := checkVela2InputCoverage(map[string]json.RawMessage{"q": json.RawMessage(`{"type":"noul"}`)}, response); err == nil {
		t.Fatal("missing strict request flag passed")
	}
	delete(complete, "input_coverage")
	if err := checkVela2InputCoverage(questions, response); err == nil {
		t.Fatal("missing worker proof passed")
	}
}

func TestVela2PublishedRecordPreservesModelLabelsAndNumbers(t *testing.T) {
	raw := map[string]interface{}{"answers": map[string]interface{}{"q": map[string]interface{}{"type": "choice", "input_coverage": "complete", "probabilities": map[string]interface{}{"input_coverage": 0.12345678}}}}
	got := recordedAnswer(raw)
	want := map[string]interface{}{"answers": map[string]interface{}{"q": map[string]interface{}{"type": "choice", "probabilities": map[string]interface{}{"input_coverage": 0.1234568}}}}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("record=%v want=%v", got, want)
	}
	if raw["answers"].(map[string]interface{})["q"].(map[string]interface{})["input_coverage"] != "complete" {
		t.Fatal("recording mutated the observed response")
	}
}
