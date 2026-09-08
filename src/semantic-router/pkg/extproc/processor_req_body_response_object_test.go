package extproc

import (
	"bytes"
	"encoding/json"
	"errors"
	"testing"

	typev3 "github.com/envoyproxy/go-control-plane/envoy/type/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/responseapi"
)

func TestSameFormatResponsesBodyPublishesRouterOwnedID(t *testing.T) {
	router := &OpenAIRouter{}
	ctx := &RequestContext{
		SourceFormat: llmprotocol.OpenAIResponsesV1,
		TargetFormat: llmprotocol.OpenAIResponsesV1,
		TraceContext: t.Context(),
		ResponseObjectState: &ResponseObjectState{
			GeneratedResponseID: "resp_router_owned",
		},
	}
	response := router.handleNonStreamingResponseBody(
		extProcResponseFixture(llmprotocol.OpenAIResponsesV1), ctx, 0,
	)
	mutation := response.GetResponseBody().GetResponse().GetBodyMutation()
	if mutation == nil || !bytes.Contains(mutation.GetBody(), []byte(`"id":"resp_router_owned"`)) {
		t.Fatalf("same-format response did not publish Router-owned ID: %+v", mutation)
	}
}

func TestSameFormatResponsesStreamPublishesRouterOwnedID(t *testing.T) {
	router := &OpenAIRouter{}
	ctx := &RequestContext{
		SourceFormat: llmprotocol.OpenAIResponsesV1,
		TargetFormat: llmprotocol.OpenAIResponsesV1,
		RequestModel: "public-model",
		TraceContext: t.Context(),
		ResponseObjectState: &ResponseObjectState{
			GeneratedResponseID: "resp_router_owned",
		},
	}
	response := router.handleSemanticStreamingResponseBody(
		extProcStreamFixture(llmprotocol.OpenAIResponsesV1), true, ctx,
	)
	mutation := response.GetResponseBody().GetResponse().GetBodyMutation()
	if mutation == nil || !bytes.Contains(mutation.GetBody(), []byte(`"id":"resp_router_owned"`)) {
		t.Fatalf("same-format stream did not publish Router-owned ID: %+v", mutation)
	}
	if ctx.SemanticResponse == nil {
		t.Fatal("same-format stream did not produce a semantic response")
	}
	if ctx.SemanticResponse.ID != "resp_router_owned" {
		t.Fatalf("stored semantic response ID = %q, want Router-owned ID", ctx.SemanticResponse.ID)
	}
}

func TestMaterializeResponseObjectContextRemovesRouterControls(t *testing.T) {
	store := false
	autoStore := true
	request := &llmprotocol.Request{
		Generation:         1,
		PreviousResponseID: "resp_previous",
		ConversationID:     "conv_one",
		Store:              &store,
		AutoStore:          &autoStore,
		Messages: []llmprotocol.Message{{
			Role:    llmprotocol.RoleUser,
			Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "current"}},
		}},
	}
	ctx := &RequestContext{ResponseObjectState: &ResponseObjectState{}}

	changed, err := (&OpenAIRouter{}).materializeResponseObjectContext(request, ctx)
	if err != nil {
		t.Fatal(err)
	}
	if !changed {
		t.Fatal("expected Router response-object controls to change the provider request")
	}
	if request.PreviousResponseID != "" || request.ConversationID != "" || request.Store != nil || request.AutoStore != nil {
		t.Fatalf("Router response-object controls leaked to provider request: %+v", request)
	}
	changed, err = (&OpenAIRouter{}).materializeResponseObjectContext(request, ctx)
	if err != nil {
		t.Fatal(err)
	}
	if changed {
		t.Fatal("response-object context was applied twice")
	}
}

func TestPrepareObjectStateCreatesUniquePublicResponseIdentity(t *testing.T) {
	filter := &ResponseAPIFilter{}
	first, err := filter.PrepareObjectState(t.Context(), llmprotocol.Request{}, nil)
	if err != nil {
		t.Fatal(err)
	}
	second, err := filter.PrepareObjectState(t.Context(), llmprotocol.Request{}, nil)
	if err != nil {
		t.Fatal(err)
	}

	if !responseapi.IsValidResponseID(first.GeneratedResponseID) {
		t.Fatalf("first response ID = %q", first.GeneratedResponseID)
	}
	if !responseapi.IsValidResponseID(second.GeneratedResponseID) {
		t.Fatalf("second response ID = %q", second.GeneratedResponseID)
	}
	if first.GeneratedResponseID == second.GeneratedResponseID {
		t.Fatalf("response IDs must be unique: %q", first.GeneratedResponseID)
	}
}

func TestPrepareObjectStateRejectsUnavailableAndMissingPreviousResponse(t *testing.T) {
	request := llmprotocol.Request{PreviousResponseID: "resp_missing"}
	_, err := (&ResponseAPIFilter{}).PrepareObjectState(t.Context(), request, nil)
	assertResponseObjectStateError(
		t, err, llmprotocol.ErrorUpstreamUnavailable, "response_history_unavailable",
	)

	filter := NewResponseAPIFilter(NewMockResponseStore())
	_, err = filter.PrepareObjectState(t.Context(), request, nil)
	assertResponseObjectStateError(t, err, llmprotocol.ErrorNotFound, "previous_response_not_found")
}

// TestPrepareObjectStateConversationMembershipIsStrictlyExplicit covers the
// three ConversationID/SessionTrackingID sources: explicit conversation_id
// (external membership + a namespaced tracking id), previous_response_id
// only (lineage: no membership, but tracking continuity is preserved via a
// separate internal key), and neither field (standalone: no membership, a
// fresh tracking id). Conversation and lineage must never blend: a
// previous_response_id-only continuation of a conversation-tagged response
// must not inherit that conversation's ID (see issue #2999 discussion on
// why chain-only forks must not silently join a Conversation's history).
func TestPrepareObjectStateConversationMembershipIsStrictlyExplicit(t *testing.T) {
	t.Run("explicit conversation_id is external membership and namespaces the tracking id", func(t *testing.T) {
		filter := NewResponseAPIFilter(NewMockResponseStore())
		state, err := filter.PrepareObjectState(t.Context(), llmprotocol.Request{ConversationID: "conv_y"}, nil)
		if err != nil {
			t.Fatal(err)
		}
		if state.ConversationID != "conv_y" {
			t.Fatalf("ConversationID = %q, want explicit value", state.ConversationID)
		}
		if state.SessionTrackingID != "respapi:conversation:conv_y" {
			t.Fatalf("SessionTrackingID = %q", state.SessionTrackingID)
		}
	})

	t.Run("previous_response_id only does not inherit the parent's conversation", func(t *testing.T) {
		store := NewMockResponseStore()
		parent := &responseapi.StoredResponse{ID: "resp_parent", ConversationID: "conv_x"}
		if err := store.StoreResponse(t.Context(), parent); err != nil {
			t.Fatal(err)
		}
		filter := NewResponseAPIFilter(store)

		state, err := filter.PrepareObjectState(
			t.Context(), llmprotocol.Request{PreviousResponseID: "resp_parent"}, nil,
		)
		if err != nil {
			t.Fatal(err)
		}
		if state.ConversationID != "" {
			t.Fatalf("ConversationID = %q, want empty: lineage must not imply conversation membership", state.ConversationID)
		}
		if state.SessionTrackingID != "respapi:lineage:resp_parent" {
			t.Fatalf("SessionTrackingID = %q", state.SessionTrackingID)
		}
	})

	t.Run("neither field is a standalone turn with no membership but a fresh tracking id", func(t *testing.T) {
		filter := NewResponseAPIFilter(NewMockResponseStore())
		state, err := filter.PrepareObjectState(t.Context(), llmprotocol.Request{}, nil)
		if err != nil {
			t.Fatal(err)
		}
		if state.ConversationID != "" {
			t.Fatalf("ConversationID = %q, want empty for a standalone turn", state.ConversationID)
		}
		if state.SessionTrackingID != sessionTrackingStandalonePrefix+state.GeneratedResponseID {
			t.Fatalf("SessionTrackingID = %q, want standalone prefix over the generated response id", state.SessionTrackingID)
		}
	})
}

func TestPrepareProtocolRequestReturnsTypedPreviousResponseFailures(t *testing.T) {
	body := []byte(`{"model":"model-a","input":"continue","previous_response_id":"resp_missing"}`)
	tests := []struct {
		name       string
		filter     *ResponseAPIFilter
		wantStatus typev3.StatusCode
		wantText   string
	}{
		{
			name: "store unavailable", wantStatus: typev3.StatusCode_ServiceUnavailable,
			wantText: "retained response history is unavailable",
		},
		{
			name: "response missing", filter: NewResponseAPIFilter(NewMockResponseStore()),
			wantStatus: typev3.StatusCode_NotFound, wantText: "previous response was not found",
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			router := &OpenAIRouter{ResponseAPIFilter: test.filter}
			ctx := &RequestContext{SourceFormat: llmprotocol.OpenAIResponsesV1, TraceContext: t.Context()}
			request, immediate := router.prepareProtocolRequest(body, ctx)
			if request != nil || immediate == nil {
				t.Fatalf("request=%+v immediate=%+v", request, immediate)
			}
			immediate = router.encodeImmediateResponseForClient(immediate, ctx)
			assertBodyImmediateErrorResponse(t, immediate, test.wantStatus, test.wantText)
		})
	}
}

func assertResponseObjectStateError(
	t *testing.T,
	err error,
	category llmprotocol.ErrorCategory,
	code string,
) {
	t.Helper()
	var protocolError *llmprotocol.ProtocolError
	if !errors.As(err, &protocolError) || protocolError.Category != category || protocolError.Code != code {
		t.Fatalf("response-object error = %T %v, want %s/%s", err, err, category, code)
	}
}

func TestParseResponseAPIInputItemsRetainsOfficialToolAndReasoningVariants(t *testing.T) {
	items := parseResponseAPIInputItems(json.RawMessage(`[
		{"type":"function_call","id":"item_call","call_id":"call_1","name":"lookup","arguments":"{\"q\":\"vLLM\"}"},
		{"type":"function_call_output","id":"item_result","call_id":"call_1","output":[{"type":"input_text","text":"result"}]},
		{"type":"reasoning","id":"item_reasoning","summary":[{"type":"summary_text","text":"inspect"}],"content":[{"type":"reasoning_text","text":"details"}]}
	]`))
	if len(items) != 3 {
		t.Fatalf("input items = %d, want 3: %+v", len(items), items)
	}
	if items[0].CallID != "call_1" || items[0].Name != "lookup" || items[0].Arguments != `{"q":"vLLM"}` {
		t.Fatalf("function call snapshot = %+v", items[0])
	}
	if items[1].CallID != "call_1" || string(items[1].Output) != `[{"type":"input_text","text":"result"}]` {
		t.Fatalf("function result snapshot = %+v", items[1])
	}
	if string(items[2].Summary) != `[{"type":"summary_text","text":"inspect"}]` ||
		string(items[2].Content) != `[{"type":"reasoning_text","text":"details"}]` {
		t.Fatalf("reasoning snapshot = %+v", items[2])
	}
	for _, item := range items {
		if item.Status != responseapi.StatusCompleted {
			t.Fatalf("snapshot status = %q", item.Status)
		}
	}
}

func TestMaterializeResponseObjectContextPrependsHistoryWithoutInheritingInstructions(t *testing.T) {
	input, err := json.Marshal("first turn")
	if err != nil {
		t.Fatal(err)
	}
	request := &llmprotocol.Request{Messages: []llmprotocol.Message{{
		Role:    llmprotocol.RoleUser,
		Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "second turn"}},
	}}}
	ctx := &RequestContext{ResponseObjectState: &ResponseObjectState{
		ConversationHistory: []*responseapi.StoredResponse{{
			Input:        []responseapi.InputItem{{Type: "message", Role: "user", Content: input}},
			OutputText:   "first answer",
			Instructions: "Keep every answer concise.",
		}},
	}}

	changed, err := (&OpenAIRouter{}).materializeResponseObjectContext(request, ctx)
	if err != nil {
		t.Fatal(err)
	}
	if !changed {
		t.Fatal("expected retained response context to change the provider request")
	}
	if len(request.Messages) != 3 {
		t.Fatalf("materialized messages = %d, want 3: %+v", len(request.Messages), request.Messages)
	}
	got := []string{
		request.Messages[0].Content[0].Text,
		request.Messages[1].Content[0].Text,
		request.Messages[2].Content[0].Text,
	}
	want := []string{"first turn", "first answer", "second turn"}
	for index := range want {
		if got[index] != want[index] {
			t.Fatalf("message %d = %q, want %q", index, got[index], want[index])
		}
	}
	if len(request.Instructions) != 0 {
		t.Fatalf("previous-response instructions must not be inherited: %+v", request.Instructions)
	}
}

func TestMaterializeResponseObjectContextPreservesRetainedToolAndReasoningLifecycle(t *testing.T) {
	request := &llmprotocol.Request{
		Generation:         1,
		PreviousResponseID: "resp_tool_result",
		Messages: []llmprotocol.Message{{
			Role:    llmprotocol.RoleUser,
			Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "summarize it"}},
		}},
	}
	ctx := &RequestContext{ResponseObjectState: &ResponseObjectState{
		ConversationHistory: []*responseapi.StoredResponse{
			{
				Input: []responseapi.InputItem{{
					Type: responseapi.ItemTypeMessage, Role: responseapi.RoleUser,
					Content: json.RawMessage(`[{"type":"input_text","text":"check Paris"}]`),
				}},
				Output: []responseapi.OutputItem{{
					Type: responseapi.ItemTypeFunctionCall, ID: "item_call", CallID: "call_weather",
					Name: "lookup_weather", Arguments: `{"city":"Paris"}`,
				}},
			},
			{
				Input: []responseapi.InputItem{{
					Type: responseapi.ItemTypeFunctionCallOutput, ID: "item_result",
					CallID: "call_weather", Output: json.RawMessage(`"sunny"`),
				}},
				Output: []responseapi.OutputItem{
					{
						Type: responseapi.ItemTypeReasoning, ID: "item_reasoning",
						Summary: []responseapi.ContentPart{{Type: "summary_text", Text: "Use the observation."}},
					},
					{
						Type: responseapi.ItemTypeMessage, ID: "item_answer", Role: responseapi.RoleAssistant,
						Content: []responseapi.ContentPart{{Type: responseapi.ContentTypeOutputText, Text: "Paris is sunny."}},
					},
				},
			},
		},
	}}

	changed, err := (&OpenAIRouter{}).materializeResponseObjectContext(request, ctx)
	if err != nil {
		t.Fatal(err)
	}
	if !changed || request.PreviousResponseID != "" {
		t.Fatalf("changed=%v previous_response_id=%q", changed, request.PreviousResponseID)
	}
	if len(request.Messages) != 6 {
		t.Fatalf("materialized messages = %d, want 6: %+v", len(request.Messages), request.Messages)
	}
	assertRetainedToolLifecycle(t, request.Messages)
	assertRetainedReasoningLifecycle(t, request.Messages)
}

func assertRetainedToolLifecycle(t *testing.T, messages []llmprotocol.Message) {
	t.Helper()
	call := messages[1].Content[0].ToolCall
	if call == nil || call.ID != "call_weather" || call.Name != "lookup_weather" || call.Arguments != `{"city":"Paris"}` {
		t.Fatalf("retained tool call = %+v", call)
	}
	result := messages[2].Content[0].ToolResult
	if result == nil || result.CallID != "call_weather" || result.DeferredLink ||
		len(result.Content) != 1 || result.Content[0].Text != "sunny" {
		t.Fatalf("retained tool result = %+v", result)
	}
}

func assertRetainedReasoningLifecycle(t *testing.T, messages []llmprotocol.Message) {
	t.Helper()
	reasoning := messages[3].Content[0]
	if reasoning.Kind != llmprotocol.ContentReasoning ||
		reasoning.Reasoning != llmprotocol.ReasoningScopeSummary ||
		reasoning.Text != "Use the observation." {
		t.Fatalf("retained reasoning = %+v", reasoning)
	}
	if got := messages[4].Content[0].Text; got != "Paris is sunny." {
		t.Fatalf("retained answer = %q", got)
	}
	if got := messages[5].Content[0].Text; got != "summarize it" {
		t.Fatalf("current message = %q", got)
	}
}

func TestMaterializeResponseObjectContextRejectsBrokenRetainedToolLifecycleAtomically(t *testing.T) {
	request := &llmprotocol.Request{
		Generation:         1,
		PreviousResponseID: "resp_broken",
		Messages: []llmprotocol.Message{{
			Role:    llmprotocol.RoleUser,
			Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "continue"}},
		}},
	}
	ctx := &RequestContext{ResponseObjectState: &ResponseObjectState{
		ConversationHistory: []*responseapi.StoredResponse{{
			Input: []responseapi.InputItem{{
				Type:   responseapi.ItemTypeFunctionCallOutput,
				CallID: "missing_call", Output: json.RawMessage(`"orphan"`),
			}},
		}},
	}}

	changed, err := (&OpenAIRouter{}).materializeResponseObjectContext(request, ctx)
	if err == nil || changed {
		t.Fatalf("changed=%v err=%v", changed, err)
	}
	if ctx.ResponseObjectState.ProviderContextApplied {
		t.Fatal("failed history materialization must remain retryable")
	}
	if request.PreviousResponseID != "resp_broken" || len(request.Messages) != 1 {
		t.Fatalf("request mutated after failed history materialization: %+v", request)
	}
}
