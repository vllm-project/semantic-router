//go:build !windows && cgo && (amd64 || arm64)

package instance

import (
	"encoding/json"
	"testing"
)

func TestOwnedSessionReportsActualInputSchema(t *testing.T) {
	model, err := LoadSequenceClassifier(fixture("sequence"))
	if err != nil {
		t.Fatal(err)
	}
	defer model.Close()
	info, err := model.Info()
	if err != nil {
		t.Fatal(err)
	}
	if len(info.Sessions) != 1 {
		t.Fatalf("expected one loaded session, got %+v", info.Sessions)
	}
	session := info.Sessions[0]
	if len(session.ExecutionInputs) != 0 || len(session.InputSchema) != 2 {
		t.Fatalf("actual schema missing or mixed with fixed execution contract: %+v", session)
	}
	for i, name := range []string{"input_ids", "attention_mask"} {
		input := session.InputSchema[i]
		if input.Name != name || input.Dtype != "int64" || len(input.Shape) != 2 || input.Shape[0] != -1 || input.Shape[1] != -1 {
			t.Fatalf("loaded dynamic graph declaration changed: %+v", input)
		}
	}
}

func TestSessionEvidenceDoesNotInferSchemaFromEmptyExecutionContract(t *testing.T) {
	var legacy SessionEvidence
	if err := json.Unmarshal([]byte(`{"execution_inputs":[]}`), &legacy); err != nil {
		t.Fatal(err)
	}
	if legacy.InputSchema != nil {
		t.Fatalf("missing input schema must remain unknown: %+v", legacy.InputSchema)
	}
	var current SessionEvidence
	if err := json.Unmarshal([]byte(`{"execution_inputs":[],"input_schema":[{"name":"input_ids","dtype":"int64","shape":[1,32768]}]}`), &current); err != nil {
		t.Fatal(err)
	}
	if len(current.InputSchema) != 1 || current.InputSchema[0].Shape[1] != 32768 {
		t.Fatalf("fixed graph length was lost: %+v", current.InputSchema)
	}
}
