package store

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/tasks"
)

func TestLabelDecisionReplayPreservesMissingScores(t *testing.T) {
	record := Record{JailbreakDetected: true, JailbreakDecision: &tasks.LabelDecision{Label: "jailbreak", SourceLabel: "unsafe", Categories: []string{"Jailbreak"}}}
	clone := cloneRecord(record)
	clone.JailbreakDecision.Categories[0] = "changed"
	if record.JailbreakDecision.Categories[0] != "Jailbreak" {
		t.Fatal("replay clone shares decision metadata")
	}
	encoded, err := json.Marshal(record)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(encoded), `"jailbreak_confidence"`) || !strings.Contains(string(encoded), `"jailbreak_decision"`) {
		t.Fatalf("incorrect categorical replay: %s", encoded)
	}
}
