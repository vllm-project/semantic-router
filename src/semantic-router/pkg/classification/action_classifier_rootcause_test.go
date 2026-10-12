package classification

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestClassifyActionTreatsBroadQuestionsAsExplain(t *testing.T) {
	tests := []struct {
		text string
		want string
	}{
		{text: "What is 2 + 2?", want: config.ActionExplain},
		{text: "Tell me about cellular biology", want: config.ActionExplain},
	}

	for _, tt := range tests {
		t.Run(tt.text, func(t *testing.T) {
			got := ClassifyAction(tt.text)
			if got.Action != tt.want || got.Score != 1 {
				t.Fatalf("ClassifyAction(%q) = %+v, want action %q with score 1", tt.text, got, tt.want)
			}
		})
	}
}
