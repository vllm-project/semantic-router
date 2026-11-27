package protocolcodec

import (
	"bytes"
	"context"
	"encoding/json"
	"reflect"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// The shapes Claude Code 2.1.325's WebSearch gets from Anthropic: a server
// tool call, its result, and text that cites the result.
const (
	webSearchTool       = `{"type":"web_search_20250305","name":"web_search","max_uses":8}`
	webSearchCall       = `{"type":"server_tool_use","id":"srvtoolu_1","name":"web_search","input":{"query":"current Go release"}}`
	webSearchResult     = `{"type":"web_search_tool_result","tool_use_id":"srvtoolu_1","content":[{"type":"web_search_result","title":"Go","url":"https://go.dev/doc/devel/release","encrypted_content":"EqQBCioIBxgC","page_age":null}]}`
	webSearchCitation   = `{"type":"web_search_result_location","cited_text":"go1.27.4 (released 2026-11-10)","url":"https://go.dev/doc/devel/release","title":"Go","encrypted_index":"Eo8BCioIBxgC"}`
	webSearchCitedText  = `{"type":"text","text":"Go 1.27.4 is the current release.","citations":[` + webSearchCitation + `]}`
	webSearchAnswerText = `{"type":"text","text":"Searched once."}`
)

func webSearchContent() string {
	return `[` + strings.Join([]string{webSearchCall, webSearchResult, webSearchAnswerText, webSearchCitedText}, ",") + `]`
}

func renameModel(request *llmprotocol.Request) error {
	request.Model = "routed-model"
	return nil
}

func requireJSONEqual(t *testing.T, label string, got, want any) {
	t.Helper()
	normalize := func(value any) any {
		var body []byte
		switch typed := value.(type) {
		case string:
			body = []byte(typed)
		case json.RawMessage:
			body = typed
		default:
			body, _ = json.Marshal(typed)
		}
		var decoded any
		if err := json.Unmarshal(body, &decoded); err != nil {
			t.Fatalf("%s: %v in %s", label, err, body)
		}
		return decoded
	}
	if !reflect.DeepEqual(normalize(got), normalize(want)) {
		t.Fatalf("%s changed\n got: %v\nwant: %v", label, got, want)
	}
}

// A request that declares a server tool, with an earlier turn's search in its
// history, reaches an Anthropic backend as written, and a Chat or Responses
// backend refuses it by name instead of dropping the tool.
func TestAnthropicServerToolsSurviveRequestRoundTrip(t *testing.T) {
	body := `{"model":"client-model","max_tokens":256,"stream":true,
		"tools":[` + webSearchTool + `,{"name":"lookup","input_schema":{"type":"object"}}],
		"messages":[{"role":"user","content":"Which Go release is current?"},
			{"role":"assistant","content":` + webSearchContent() + `},
			{"role":"user","content":"Thanks. And the one before it?"}]}`
	engine := NewBuiltinEngine()
	request, _, _, err := engine.DecodeRequest(llmprotocol.AnthropicMessagesV1, []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	if tool := request.Tools[0]; tool.Kind != llmprotocol.ToolKindServer || tool.Name != "web_search" {
		t.Fatalf("server tool decoded as %+v", tool)
	}
	history := request.Messages[1].Content
	if len(history) != 4 || history[0].Kind != llmprotocol.ContentServerTool || history[1].Kind != llmprotocol.ContentServerTool ||
		history[3].Kind != llmprotocol.ContentText || len(history[3].Raw) == 0 {
		t.Fatalf("history decoded as %+v", history)
	}

	result, err := engine.TranslateRequest(llmprotocol.AnthropicMessagesV1, llmprotocol.AnthropicMessagesV1, []byte(body), renameModel)
	if err != nil {
		t.Fatal(err)
	}
	var sent struct {
		Model    string            `json:"model"`
		Tools    json.RawMessage   `json:"tools"`
		Messages []json.RawMessage `json:"messages"`
	}
	if err = json.Unmarshal(result.Body, &sent); err != nil || sent.Model != "routed-model" {
		t.Fatalf("re-encoded request %s: %v", result.Body, err)
	}
	requireJSONEqual(t, "tools", sent.Tools, `[`+webSearchTool+`,{"name":"lookup","input_schema":{"type":"object"}}]`)
	requireJSONEqual(t, "search turn", sent.Messages[1], `{"role":"assistant","content":`+webSearchContent()+`}`)

	for _, target := range []llmprotocol.WireFormat{llmprotocol.OpenAIChatV1, llmprotocol.OpenAIResponsesV1} {
		_, err := engine.TranslateRequest(llmprotocol.AnthropicMessagesV1, target, []byte(body), renameModel)
		assertProtocolError(t, err, llmprotocol.ErrorUnsupportedFeature, "unsupported_capability")
		if !strings.Contains(err.Error(), "server_tools") {
			t.Fatalf("%s refusal does not name server tools: %v", target, err)
		}
	}
}

func TestAnthropicServerToolNeedsAName(t *testing.T) {
	body := `{"model":"m","max_tokens":16,"messages":[{"role":"user","content":"hi"}],"tools":[{"type":"web_search_20250305"}]}`
	_, _, _, err := NewBuiltinEngine().DecodeRequest(llmprotocol.AnthropicMessagesV1, []byte(body))
	assertProtocolError(t, err, llmprotocol.ErrorInvalidRequest, "invalid_tool")
}

func webSearchResponse() string {
	return `{"id":"msg_1","type":"message","role":"assistant","model":"claude-sonnet-5-5","content":` + webSearchContent() + `,
		"stop_reason":"end_turn","stop_sequence":null,
		"usage":{"input_tokens":2908,"output_tokens":57,"server_tool_use":{"web_search_requests":1,"web_fetch_requests":0}}}`
}

// A buffered reply keeps the search's blocks and citations when the Router
// re-encodes it for an Anthropic client.
func TestAnthropicServerToolsSurviveBufferedResponse(t *testing.T) {
	engine := NewBuiltinEngine()
	rename := func(response *llmprotocol.Response) error {
		response.Model = "public-model"
		return nil
	}
	result, err := engine.TranslateResponse(llmprotocol.AnthropicMessagesV1, llmprotocol.AnthropicMessagesV1, []byte(webSearchResponse()), rename)
	if err != nil {
		t.Fatal(err)
	}
	var sent struct {
		Model   string          `json:"model"`
		Content json.RawMessage `json:"content"`
	}
	if err = json.Unmarshal(result.Body, &sent); err != nil || sent.Model != "public-model" {
		t.Fatalf("re-encoded reply %s: %v", result.Body, err)
	}
	requireJSONEqual(t, "content", sent.Content, webSearchContent())
	_, err = engine.TranslateResponse(llmprotocol.AnthropicMessagesV1, llmprotocol.OpenAIChatV1, []byte(webSearchResponse()), nil)
	assertProtocolError(t, err, llmprotocol.ErrorUnsupportedFeature, "unsupported_capability")
}

// webSearchStream is the stream Anthropic sends for webSearchResponse, with
// the query split over deltas and the citation in a citations_delta.
func webSearchStream(t *testing.T) []byte {
	t.Helper()
	events := []string{
		`{"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude-sonnet-5-5","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":2908,"output_tokens":16}}}`,
		`{"type":"content_block_start","index":0,"content_block":{"type":"server_tool_use","id":"srvtoolu_1","name":"web_search","input":{}}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":""}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"query\": \"current Go"}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":" release\"}"}}`,
		`{"type":"content_block_stop","index":0}`,
		`{"type":"content_block_start","index":1,"content_block":` + webSearchResult + `}`,
		`{"type":"content_block_stop","index":1}`,
		`{"type":"content_block_start","index":2,"content_block":{"type":"text","text":""}}`,
		`{"type":"content_block_delta","index":2,"delta":{"type":"text_delta","text":"Searched once."}}`,
		`{"type":"content_block_stop","index":2}`,
		`{"type":"content_block_start","index":3,"content_block":{"citations":[],"type":"text","text":""}}`,
		`{"type":"content_block_delta","index":3,"delta":{"type":"citations_delta","citation":` + webSearchCitation + `}}`,
		`{"type":"content_block_delta","index":3,"delta":{"type":"text_delta","text":"Go 1.27.4 is the current release."}}`,
		`{"type":"content_block_stop","index":3}`,
		`{"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":57,"server_tool_use":{"web_search_requests":1,"web_fetch_requests":0}}}`,
		`{"type":"message_stop"}`,
	}
	var body bytes.Buffer
	for _, event := range events {
		var typed struct {
			Type string `json:"type"`
		}
		if err := json.Unmarshal([]byte(event), &typed); err != nil {
			t.Fatalf("fixture %s: %v", event, err)
		}
		body.WriteString("event: " + typed.Type + "\ndata: " + event + "\n\n")
	}
	return body.Bytes()
}

// replayAnthropicStream rebuilds the content a client assembles from a
// Messages stream, as the SDKs do, and returns it with the raw starts.
func replayAnthropicStream(t *testing.T, stream []byte) ([]map[string]any, []string) {
	t.Helper()
	var blocks []map[string]any
	var starts []string
	inputs := map[int]string{}
	for _, frame := range strings.Split(string(stream), "\n\n") {
		_, data, found := strings.Cut(frame, "data: ")
		if !found {
			continue
		}
		var event struct {
			Type         string          `json:"type"`
			Index        int             `json:"index"`
			ContentBlock json.RawMessage `json:"content_block"`
			Delta        struct {
				Type        string          `json:"type"`
				Text        string          `json:"text"`
				PartialJSON string          `json:"partial_json"`
				Citation    json.RawMessage `json:"citation"`
			} `json:"delta"`
		}
		if err := json.Unmarshal([]byte(data), &event); err != nil {
			t.Fatalf("output frame %q: %v", data, err)
		}
		switch event.Type {
		case "content_block_start":
			var block map[string]any
			_ = json.Unmarshal(event.ContentBlock, &block)
			if event.Index != len(blocks) {
				t.Fatalf("block %d started at index %d", len(blocks), event.Index)
			}
			blocks, starts = append(blocks, block), append(starts, string(event.ContentBlock))
		case "content_block_delta":
			block := blocks[event.Index]
			switch event.Delta.Type {
			case "text_delta":
				block["text"] = block["text"].(string) + event.Delta.Text
			case "input_json_delta":
				inputs[event.Index] += event.Delta.PartialJSON
			case "citations_delta":
				var citation any
				_ = json.Unmarshal(event.Delta.Citation, &citation)
				existing, _ := block["citations"].([]any)
				block["citations"] = append(existing, citation)
			}
		}
	}
	for index, input := range inputs {
		var decoded any
		if err := json.Unmarshal([]byte(input), &decoded); err != nil {
			t.Fatalf("block %d input %q: %v", index, input, err)
		}
		blocks[index]["input"] = decoded
	}
	return blocks, starts
}

// A streamed search reaches an Anthropic client as Anthropic streamed it: the
// call with its query, the result, and text with its citation, at the same
// indexes. Buffered settlement sees the same blocks.
func TestAnthropicServerToolsSurviveStream(t *testing.T) {
	engine := NewBuiltinEngine()
	stream, err := engine.NewStream(llmprotocol.AnthropicMessagesV1, llmprotocol.AnthropicMessagesV1,
		llmprotocol.StreamContext{Context: context.Background(), PublicModel: "public-model"})
	if err != nil {
		t.Fatal(err)
	}
	frames, _, _, err := stream.Push(webSearchStream(t))
	if err != nil {
		t.Fatal(err)
	}
	final, _, _, err := stream.Finalize(nil)
	if err != nil {
		t.Fatal(err)
	}
	output := bytes.Join(append(frames, final...), nil)
	blocks, starts := replayAnthropicStream(t, output)
	requireJSONEqual(t, "streamed content", blocks, webSearchContent())
	if !strings.Contains(starts[0], `"input":{}`) || !strings.Contains(starts[3], `"citations":[]`) {
		t.Fatalf("block starts differ from Anthropic's: %v", starts)
	}

	response, _, err := engine.DecodeResponseStream(llmprotocol.AnthropicMessagesV1, webSearchStream(t),
		llmprotocol.StreamContext{Context: context.Background(), PublicModel: "public-model"})
	if err != nil {
		t.Fatal(err)
	}
	var settled []llmprotocol.Content
	for _, item := range response.Output {
		settled = append(settled, item.Content...)
	}
	encoded, err := encodeAnthropicContent(settled)
	if err != nil {
		t.Fatal(err)
	}
	requireJSONEqual(t, "settled content", encoded, webSearchContent())

	cross, err := engine.NewStream(llmprotocol.AnthropicMessagesV1, llmprotocol.OpenAIChatV1,
		llmprotocol.StreamContext{Context: context.Background(), PublicModel: "public-model"})
	if err != nil {
		t.Fatal(err)
	}
	_, _, _, err = cross.Push(webSearchStream(t))
	assertProtocolError(t, err, llmprotocol.ErrorUnsupportedFeature, "unsupported_capability")
}
