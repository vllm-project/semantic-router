package modelservice

import (
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice/api"
	"testing"
)

func TestDecisionMissingValueNeverDecodesAsZero(t *testing.T) {
	for _, kind := range []string{"noul", "score", "choice"} {
		got := decodeAnswer(api.Answer{Type: &kind})
		if got.Error != "missing_answer_value" {
			t.Fatalf("%s missing value became successful zero: %+v", kind, got)
		}
	}
	kind, zero := "noul", 0.0
	got := decodeAnswer(api.Answer{Type: &kind, Noul: &zero})
	if got.Error != "" || got.Noul != 0 {
		t.Fatal("explicit zero rejected")
	}
}
