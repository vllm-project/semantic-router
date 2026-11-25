package extproc

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/services"
)

// Preview and live routing estimate the same agent history with different
// walkers, and each subtracts its own reserves, so both serve a history whose
// estimate exceeds the only window because of tool-call arguments and results.
func TestPreviewAndRuntimeServeHistoriesAboveTheWindowOnlyByReserves(t *testing.T) {
	router := routingTestRouter("worker")
	params := router.Config.ModelConfig["worker"]
	params.ContextWindowSize = 5000
	router.Config.ModelConfig["worker"] = params
	router.Config.Decisions = []config.Decision{{
		Name: "default", ModelRefs: []config.ModelRef{{Model: "worker"}},
	}}
	classifier, err := classification.NewClassifier(router.Config)
	require.NoError(t, err)
	t.Cleanup(func() { _ = classifier.Close() })
	router.Classifier = classifier
	service := services.NewClassificationService(classifier, router.Config)
	service.SetEvalModelSelector(router)

	arguments := `{"content":"` + strings.Repeat("x", 6000) + `"}`
	output := strings.Repeat("y", 6000)
	call, err := json.Marshal(map[string]any{
		"id": "call_1", "type": "function",
		"function": map[string]any{"name": "write", "arguments": arguments},
	})
	require.NoError(t, err)
	content, err := json.Marshal(output)
	require.NoError(t, err)
	preview, err := service.ClassifyIntentForEval(context.Background(), services.IntentRequest{
		Model: "vllm-sr/auto",
		Messages: []services.IntentMessage{
			{Role: "user", Content: json.RawMessage(`"write the file"`)},
			{Role: "assistant", Content: json.RawMessage(`null`), ToolCalls: []json.RawMessage{call}},
			{Role: "tool", ToolCallID: "call_1", Content: content},
		},
	})
	require.NoError(t, err)
	require.Equal(t, services.EvalSelectionSelected, preview.SelectionStatus, preview.SelectionReason)
	require.Equal(t, "worker", preview.SelectedModel)

	request := testNeutralRequest("vllm-sr/auto", "write the file")
	request.Messages = append(request.Messages,
		llmprotocol.Message{Role: llmprotocol.RoleAssistant, Content: []llmprotocol.Content{{
			Kind: llmprotocol.ContentToolCall, ToolCall: &llmprotocol.ToolCall{ID: "call_1", Name: "write", Arguments: arguments},
		}}},
		llmprotocol.Message{Role: llmprotocol.RoleTool, Content: []llmprotocol.Content{{
			Kind: llmprotocol.ContentToolResult, ToolResult: &llmprotocol.ToolResult{
				CallID: "call_1", Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: output}},
			},
		}}},
	)
	ctx := routingTestContext(llmprotocol.OpenAIChatV1, request)
	encoded, err := protocolcodec.NewBuiltinEngine().EncodeRequest(llmprotocol.OpenAIChatV1, *request, llmprotocol.Envelope{})
	require.NoError(t, err)
	stream := NewMockStream(nil)
	require.NoError(t, router.processRequestBody(stream, &ext_proc.ProcessingRequest_RequestBody{
		RequestBody: &ext_proc.HttpBody{Body: encoded.Body, EndOfStream: true},
	}, ctx))
	require.Len(t, stream.Responses, 1)
	require.Nil(t, stream.Responses[0].GetImmediateResponse(), "live routing must serve what preview selects")
	require.Equal(t, "worker", ctx.VSRSelectedModel)
}

// Image reserves can exclude the only model that reads images while a text
// model still fits; the fallback counts candidates that can serve the request,
// so the vision model is kept rather than the request refused.
func TestReserveFallbackKeepsTheModelThatCanServeTheRequest(t *testing.T) {
	router := routingTestRouter("text-only")
	text := router.Config.ModelConfig["text-only"]
	vision := text
	text.ContextWindowSize, text.Capabilities = 32_768, []string{"chat"}
	vision.ContextWindowSize, vision.Capabilities = 8_192, []string{"chat", "vision"}
	router.Config.ModelConfig["text-only"], router.Config.ModelConfig["vision"] = text, vision
	router.ModelSelector = selection.NewRegistry()
	router.ModelSelector.Register(selection.MethodStatic, selection.NewStaticSelector(nil))
	router.Config.Decisions = []config.Decision{{
		Name: "default", ModelRefs: []config.ModelRef{{Model: "text-only"}, {Model: "vision"}},
		Algorithm: &config.AlgorithmConfig{Type: config.DecisionAlgorithmStatic},
	}}
	classifier, err := classification.NewClassifier(router.Config)
	require.NoError(t, err)
	t.Cleanup(func() { _ = classifier.Close() })
	router.Classifier = classifier

	request := testNeutralRequest("vllm-sr/auto", "compare these screenshots")
	for range 2 {
		request.Messages[0].Content = append(request.Messages[0].Content,
			llmprotocol.Content{Kind: llmprotocol.ContentImage, URL: "https://example.invalid/image"})
	}
	ctx := routingTestContext(llmprotocol.OpenAIChatV1, request)
	encoded, err := protocolcodec.NewBuiltinEngine().EncodeRequest(llmprotocol.OpenAIChatV1, *request, llmprotocol.Envelope{})
	require.NoError(t, err)
	stream := NewMockStream(nil)
	require.NoError(t, router.processRequestBody(stream, &ext_proc.ProcessingRequest_RequestBody{
		RequestBody: &ext_proc.HttpBody{Body: encoded.Body, EndOfStream: true},
	}, ctx))
	require.Len(t, stream.Responses, 1)
	require.Nil(t, stream.Responses[0].GetImmediateResponse(), "the vision model must serve the request")
	require.Equal(t, "vision", ctx.VSRSelectedModel)

	service := services.NewClassificationService(classifier, router.Config)
	service.SetEvalModelSelector(router)
	image := `{"type":"image_url","image_url":{"url":"https://example.invalid/image"}}`
	preview, err := service.ClassifyIntentForEval(context.Background(), services.IntentRequest{
		Model: "vllm-sr/auto",
		Messages: []services.IntentMessage{{Role: "user", Content: json.RawMessage(
			`[{"type":"text","text":"compare these screenshots"},` + image + `,` + image + `]`)}},
	})
	require.NoError(t, err)
	// Preview keeps the vision candidate rather than refusing; its dry-run
	// selection does not filter by capability, with or without reserves.
	require.Equal(t, services.EvalSelectionSelected, preview.SelectionStatus, preview.SelectionReason)
}

// With known limits, live routing and preview serve a request whose estimate
// exceeds the only window because of an image reserve, keep a candidate the
// full estimate fits when there is one, and refuse a request that does not
// fit without reserves either.
func TestKnownLimitsRefuseOnlyWithoutReserves(t *testing.T) {
	for _, test := range []struct {
		name    string
		refs    []string
		text    string
		want    string
		dropped bool
	}{
		{"fits only without reserves", []string{"small"}, "describe this screenshot", "small", true},
		{"the full estimate fits a larger model", []string{"small", "large"}, "describe this screenshot", "large", false},
		{"too long without reserves", []string{"small"}, strings.Repeat("describe ", 5000), "", false},
	} {
		t.Run(test.name, func(t *testing.T) {
			router := routingTestRouter("small")
			small := router.Config.ModelConfig["small"]
			small.Capabilities, small.ContextWindowSize, small.MaxOutputTokens = []string{"chat", "vision"}, 8_192, 1_024
			large := small
			large.ContextWindowSize = 32_768
			router.Config.ModelConfig["small"], router.Config.ModelConfig["large"] = small, large
			router.Config.CandidateRequirements = &config.CandidateRequirements{Capabilities: config.CandidateCapabilitiesDeclared, Context: config.CandidateContextKnownLimits}
			router.ModelSelector = selection.NewRegistry()
			router.ModelSelector.Register(selection.MethodStatic, selection.NewStaticSelector(nil))
			refs := make([]config.ModelRef, 0, len(test.refs))
			for _, model := range test.refs {
				refs = append(refs, config.ModelRef{Model: model})
			}
			router.Config.Decisions = []config.Decision{{
				Name: "default", ModelRefs: refs,
				Algorithm: &config.AlgorithmConfig{Type: config.DecisionAlgorithmStatic},
			}}
			classifier, err := classification.NewClassifier(router.Config)
			require.NoError(t, err)
			t.Cleanup(func() { _ = classifier.Close() })
			router.Classifier = classifier

			request := testNeutralRequest("vllm-sr/auto", test.text)
			request.Messages[0].Content = append(request.Messages[0].Content,
				llmprotocol.Content{Kind: llmprotocol.ContentImage, URL: "https://example.invalid/image"})
			request.Sampling.MaxOutputTokens = llmprotocol.Int64(512)
			ctx := routingTestContext(llmprotocol.OpenAIChatV1, request)
			encoded, err := protocolcodec.NewBuiltinEngine().EncodeRequest(llmprotocol.OpenAIChatV1, *request, llmprotocol.Envelope{})
			require.NoError(t, err)
			stream := NewMockStream(nil)
			require.NoError(t, router.processRequestBody(stream, &ext_proc.ProcessingRequest_RequestBody{
				RequestBody: &ext_proc.HttpBody{Body: encoded.Body, EndOfStream: true},
			}, ctx))
			require.Len(t, stream.Responses, 1)
			immediate := stream.Responses[0].GetImmediateResponse()

			service := services.NewClassificationService(classifier, router.Config)
			service.SetEvalModelSelector(router)
			content, err := json.Marshal([]map[string]any{
				{"type": "text", "text": test.text},
				{"type": "image_url", "image_url": map[string]any{"url": "https://example.invalid/image"}},
			})
			require.NoError(t, err)
			preview, err := service.ClassifyIntentForEval(context.Background(), services.IntentRequest{
				Model:     "vllm-sr/auto",
				Messages:  []services.IntentMessage{{Role: "user", Content: content}},
				MaxTokens: json.RawMessage(`512`),
			})
			require.NoError(t, err)

			if test.want == "" {
				require.NotNil(t, immediate, "a request too long without reserves must be refused")
				require.Contains(t, string(immediate.Body), "context_length_exceeded")
				require.Equal(t, services.EvalSelectionUnavailable, preview.SelectionStatus)
				return
			}
			require.Nil(t, immediate, "the request must be served")
			require.Equal(t, test.want, ctx.VSRSelectedModel)
			require.Equal(t, test.dropped, ctx.VSRContextReserveTokens == 0)
			require.Equal(t, services.EvalSelectionSelected, preview.SelectionStatus, preview.SelectionReason)
			require.Equal(t, test.want, preview.SelectedModel)
		})
	}
}

// Live selection scores candidates on the count the eligibility filters used:
// with the image reserve while the filters count it, without it once a filter
// dropped it, as preview does.
func TestLiveSelectionScoresTheCountTheFiltersUsed(t *testing.T) {
	request := testNeutralRequest("vllm-sr/auto", "describe this screenshot")
	request.Messages[0].Content = append(request.Messages[0].Content,
		llmprotocol.Content{Kind: llmprotocol.ContentImage, URL: "https://example.invalid/image"})
	full := selection.DemandForRequest(request)
	require.Positive(t, full.InputReserveTokens)
	ctx := routingTestContext(llmprotocol.OpenAIChatV1, request)
	decision := &config.Decision{Name: "default"}

	ctx.VSRContextReserveTokens = full.InputReserveTokens
	require.Equal(t, full.InputTokens, candidateSelectionDemand(ctx, decision).InputTokens)
	dropContextReserves(ctx, decision.Name)
	require.Equal(t, full.WithoutReserves().InputTokens, candidateSelectionDemand(ctx, decision).InputTokens)
}
