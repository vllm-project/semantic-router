package contextcompression

import (
	"context"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func exchangeFixture() *llmprotocol.Request {
	request := historyFixture()
	request.Messages = append([]llmprotocol.Message{
		textMessage(llmprotocol.RoleUser, "old question"),
		{Role: llmprotocol.RoleAssistant, Content: []llmprotocol.Content{
			{Kind: llmprotocol.ContentToolCall, ToolCall: &llmprotocol.ToolCall{ID: "a", Name: "lookup", Arguments: "{}"}},
			{Kind: llmprotocol.ContentToolCall, ToolCall: &llmprotocol.ToolCall{ID: "b", Name: "lookup", Arguments: "{}"}},
		}},
		{Role: llmprotocol.RoleTool, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentToolResult, ToolResult: &llmprotocol.ToolResult{CallID: "a", Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "a result"}}}}}},
		{Role: llmprotocol.RoleTool, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentToolResult, ToolResult: &llmprotocol.ToolResult{CallID: "b", Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "b result"}}}}}},
	}, request.Messages[2:]...)
	return request
}

func TestTransformationParallelToolExchangeAtomicity(t *testing.T) {
	for _, ids := range [][]int{{0, 1, 2}, {0, 1, 2, 3}} {
		request := exchangeFixture()
		ir := ParseSemanticRequest(request, Provenance{})
		err := ir.ApplySteps(context.Background(), []TransformationStep{removalStep(TransformSelectTurns, ids...)})
		if (err == nil) != (len(ids) == 4) {
			t.Fatalf("wrong exchange eligibility for %v: %v", ids, err)
		}
	}
}

func TestTransformationToolResultsDoNotStartTurns(t *testing.T) {
	request := exchangeFixture()
	request.Messages[2].Role = llmprotocol.RoleUser
	request.Messages[2].Content = append(request.Messages[2].Content, request.Messages[3].Content...)
	request.Messages = append(request.Messages[:3], request.Messages[4:]...)
	ir := ParseSemanticRequest(request, Provenance{})
	if ir.Messages[2].TurnID != 0 || len(ir.Messages[2].ExchangeIDs) != 2 {
		t.Fatalf("lost grouped results: %+v", ir.Messages[2])
	}
	if err := ir.ApplySteps(context.Background(), []TransformationStep{removalStep(TransformReset, 0, 1, 2)}); err != nil {
		t.Fatal(err)
	}
}

func TestTransformationMultimodalAndUnresolvedExchangesProtected(t *testing.T) {
	for _, multimodal := range []bool{true, false} {
		request := exchangeFixture()
		if multimodal {
			request.Messages[3].Content[0].ToolResult.Content = []llmprotocol.Content{{Kind: llmprotocol.ContentImage, URL: "https://example.com/image.png"}}
		} else {
			request.Messages[3].Content[0].ToolResult.CallID = "unresolved"
		}
		before := encoded(t, request)
		ir := ParseSemanticRequest(request, Provenance{})
		if err := ir.ApplySteps(context.Background(), []TransformationStep{removalStep(TransformReset, 0, 1, 2, 3)}); err == nil {
			t.Fatal("unsafe removal accepted")
		}
		if encoded(t, request) != before {
			t.Fatal("protected request changed")
		}
	}
}

func TestTransformationRAGOriginProtectsWholeTurn(t *testing.T) {
	request := exchangeFixture()
	ir := ParseSemanticRequest(request, Provenance{RAGToolCallIDs: map[string]struct{}{"a": {}, "b": {}}})
	for _, id := range []int{1, 2, 3} {
		if ir.Messages[id].Source != SourceRAG {
			t.Fatalf("RAG source lost: %+v", ir.Messages[id])
		}
	}
	if err := ir.ApplySteps(context.Background(), []TransformationStep{removalStep(TransformReset, 0, 1, 2, 3)}); err == nil {
		t.Fatal("injected content removed as history")
	}
}
