package extproc

import (
	"testing"

	"github.com/openai/openai-go"
	"github.com/openai/openai-go/packages/param"
)

func TestToolEmbeddingText_IncludesDescription(t *testing.T) {
	tp := openai.ChatCompletionToolParam{
		Type: "function",
		Function: openai.FunctionDefinitionParam{
			Name:        "alpha",
			Description: param.NewOpt("desc here"),
		},
	}
	if got := toolEmbeddingText(tp); got != "alpha desc here" {
		t.Fatalf("unexpected text: %q", got)
	}
}
