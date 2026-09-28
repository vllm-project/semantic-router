package extproc

import (
	"context"
	"reflect"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/contextcompression"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func TestContextPlanOriginalHistoryPrecedesEnrichment(t *testing.T) {
	request := &llmprotocol.Request{Messages: []llmprotocol.Message{
		{Role: llmprotocol.RoleUser, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "question"}}},
	}}
	ctx := &RequestContext{SemanticRequest: request}
	captureOriginalContextHistory(ctx)
	request.Messages = append(request.Messages,
		llmprotocol.Message{Role: llmprotocol.RoleAssistant, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentToolCall, ToolCall: &llmprotocol.ToolCall{ID: "rag-call", Name: "search", Arguments: "{}"}}}},
		llmprotocol.Message{Role: llmprotocol.RoleTool, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentToolResult, ToolResult: &llmprotocol.ToolResult{CallID: "rag-call", Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "document"}}}}}},
	)
	ctx.RAGToolCallIDs = map[string]struct{}{"rag-call": {}}
	memory := llmprotocol.Message{Role: llmprotocol.RoleUser, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "saved fact"}}}
	request.Messages = append([]llmprotocol.Message{memory}, request.Messages...)
	ctx.MemoryMessageIndexes = map[int]struct{}{0: {}}
	before := append([]llmprotocol.Message(nil), request.Messages...)
	if err := (&OpenAIRouter{}).applyContextTransformationPlan(ctx, request); err != nil {
		t.Fatal(err)
	}
	ir := ctx.ContextRequestIR
	if len(ir.OriginalHistory().Messages) != 1 || ir.OriginalHistory().Messages[0].Content[0].Text != "question" {
		t.Fatal("enrichment contaminated original history")
	}
	if ir.Messages[0].Source != contextcompression.SourceMemory || ir.Messages[2].Source != contextcompression.SourceRAG || ir.Messages[3].Source != contextcompression.SourceRAG {
		t.Fatal("injection provenance lost")
	}
	if !reflect.DeepEqual(before, request.Messages) {
		t.Fatal("disabled plan changed enriched request")
	}
	if len(ir.Transformations.Receipts()) != 1 {
		t.Fatal("disabled compression receipt missing")
	}
}

func TestContextPlanRunsWithoutCompressionPlugin(t *testing.T) {
	request := &llmprotocol.Request{}
	for i, role := range []llmprotocol.Role{llmprotocol.RoleUser, llmprotocol.RoleAssistant, llmprotocol.RoleUser, llmprotocol.RoleAssistant, llmprotocol.RoleUser} {
		request.Messages = append(request.Messages, llmprotocol.Message{Role: role, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: string(rune('a' + i))}}})
	}
	ctx := &RequestContext{SemanticRequest: request, ContextHistorySteps: []contextcompression.TransformationStep{{
		Kind: contextcompression.TransformSelectTurns, Enabled: true, FailureMode: contextcompression.FailureClosed,
		Propose: func(context.Context, contextcompression.TransformationView) (contextcompression.TransformationEdits, error) {
			return contextcompression.TransformationEdits{RemoveMessages: []int{0, 1}}, nil
		},
	}}}
	captureOriginalContextHistory(ctx)
	if err := (&OpenAIRouter{}).applyContextTransformationPlan(ctx, request); err != nil {
		t.Fatal(err)
	}
	if len(request.Messages) != 3 || len(ctx.ContextRequestIR.OriginalHistory().Messages) != 5 {
		t.Fatal("wrong history after selection")
	}
	receipts := ctx.ContextRequestIR.Transformations.Receipts()
	if len(receipts) != 2 || receipts[0].Kind != contextcompression.TransformSelectTurns || receipts[1].Kind != contextcompression.TransformCompress {
		t.Fatal("pipeline order lost")
	}
}
