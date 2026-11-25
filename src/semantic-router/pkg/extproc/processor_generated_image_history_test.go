package extproc

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/responseapi"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/responsestore"
)

func TestStoredGeneratedImageHistory(t *testing.T) {
	for _, mode := range []string{"text-control", "buffered-image", "streamed-image"} {
		t.Run(mode, func(t *testing.T) {
			store, err := responsestore.NewMemoryStore(responsestore.StoreConfig{Enabled: true})
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() { _ = store.Close() })
			router := &OpenAIRouter{ResponseAPIFilter: NewResponseAPIFilter(store)}
			ctx := &RequestContext{
				SourceFormat: llmprotocol.OpenAIResponsesV1, TargetFormat: llmprotocol.OpenAIResponsesV1,
				Headers: map[string]string{}, TraceContext: t.Context(), RequestID: "stored-image", RequestModel: "image-model", UpstreamStatusCode: 200,
			}
			requestBody := []byte(`{"model":"image-model","input":"Draw a cat"}`)
			image := mode != "text-control"
			if image {
				requestBody = []byte(`{"model":"image-model","input":"Draw a cat","tools":[{"type":"image_generation"}]}`)
			}
			request, immediate := router.prepareProtocolRequest(requestBody, ctx)
			if request == nil || immediate != nil || !ctx.ResponseObjectState.ShouldStore {
				t.Fatalf("default-store request rejected: request=%+v immediate=%+v", request, immediate)
			}
			if mode == "streamed-image" {
				streamingClient(ctx, llmprotocol.OpenAIResponsesV1)
				chunks := imageGenerationStream(t)
				for i, chunk := range chunks {
					router.handleSemanticStreamingResponseBody(chunk, i == len(chunks)-1, ctx)
				}
				if ctx.StreamingAborted || ctx.SemanticResponse == nil {
					t.Fatal("the original image stream did not complete")
				}
				client, encodeErr := router.encodeClientResponse(*ctx.SemanticResponse, ctx)
				if encodeErr != nil || !bytes.Contains(client, []byte(`"result":"`+streamedImage+`"`)) {
					t.Fatalf("the original client response lost its image: %s %v", client, encodeErr)
				}
			} else {
				body := extProcResponseFixture(llmprotocol.OpenAIResponsesV1)
				if image {
					body = []byte(`{"id":"resp_image","object":"response","model":"image-model","status":"completed","output":[` +
						`{"type":"image_generation_call","id":"ig_1","status":"completed","result":"` + streamedImage + `"}]}`)
				}
				if _, _, _, decodeErr := protocolcodec.NewBuiltinEngine().DecodeResponse(llmprotocol.OpenAIResponsesV1, body); decodeErr != nil {
					t.Fatalf("the source fixture is not a valid response: %v", decodeErr)
				}
				reply := router.handleNonStreamingResponseBody(body, ctx, 0, false)
				if reply.GetImmediateResponse() != nil || ctx.SemanticResponse == nil {
					t.Fatalf("the original response failed: %v", reply)
				}
				if image && !bytes.Contains(reply.GetResponseBody().GetResponse().GetBodyMutation().GetBody(), []byte(`"result":"`+streamedImage+`"`)) {
					t.Fatal("the original buffered client response lost its image")
				}
			}
			router.persistResponseObject(ctx)
			id := ctx.SemanticResponse.ID
			get, err := router.ResponseAPIFilter.HandleGetResponse(t.Context(), id)
			if err != nil || get.GetImmediateResponse().GetStatus().GetCode() != 200 {
				t.Fatalf("GET %s failed: %v %v", id, get, err)
			}
			body := get.GetImmediateResponse().GetBody()
			_, _, _, getDecodeErr := protocolcodec.NewBuiltinEngine().DecodeResponse(llmprotocol.OpenAIResponsesV1, body)
			t.Logf("GET stored response: %s; codec error: %v", body, getDecodeErr)
			if image && !bytes.Contains(body, []byte(`"result":"`+streamedImage+`"`)) {
				t.Error("stored GET dropped the generated image result that the original client received")
			}
			if getDecodeErr != nil {
				t.Errorf("the stored GET is no longer a valid Responses object: %v", getDecodeErr)
			}
			next := &RequestContext{
				SourceFormat: llmprotocol.OpenAIResponsesV1, TargetFormat: llmprotocol.OpenAIResponsesV1,
				TraceContext: t.Context(), Headers: map[string]string{},
			}
			request, immediate = router.prepareProtocolRequest([]byte(fmt.Sprintf(`{"model":"image-model","previous_response_id":%q,"input":"Make it blue"}`, id)), next)
			if request == nil || immediate != nil {
				t.Fatalf("retained turn request rejected at load: %v", immediate)
			}
			_, err = router.materializeResponseObjectContext(request, next)
			t.Logf("previous_response_id materialization: %v", err)
			if err != nil {
				t.Errorf("a continuation of a successful image response failed: %v", err)
			}
			if image && err == nil {
				assertGeneratedImageHistoryResult(t, request)
			}
		})
	}
}

func TestGeneratedImageStatelessReplayControl(t *testing.T) {
	body := []byte(`{"model":"image-model","store":false,"input":[` +
		`{"type":"image_generation_call","id":"ig_1","status":"completed","result":"` + streamedImage + `"},` +
		`{"role":"user","content":"Make it blue"}]}`)
	request, _, _, err := protocolcodec.NewBuiltinEngine().DecodeRequest(llmprotocol.OpenAIResponsesV1, body)
	if err != nil || len(request.Messages) == 0 {
		t.Fatalf("stateless replay with the intact image must work: %v", err)
	}
}

func TestStoredGeneratedImageInput(t *testing.T) {
	store, err := responsestore.NewMemoryStore(responsestore.StoreConfig{Enabled: true})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = store.Close() })
	router := &OpenAIRouter{ResponseAPIFilter: NewResponseAPIFilter(store)}
	ctx := &RequestContext{
		SourceFormat: llmprotocol.OpenAIResponsesV1, TargetFormat: llmprotocol.OpenAIResponsesV1,
		TraceContext: t.Context(), Headers: map[string]string{}, RequestModel: "image-model",
	}
	body := []byte(`{"model":"image-model","input":[` +
		`{"type":"image_generation_call","id":"ig_1","status":"completed","result":"` + streamedImage + `"},` +
		`{"role":"user","content":"Make it blue"}]}`)
	request, immediate := router.prepareProtocolRequest(body, ctx)
	if request == nil || immediate != nil {
		t.Fatalf("intact image input failed before storage: %v", immediate)
	}
	reply := router.handleNonStreamingResponseBody(extProcResponseFixture(llmprotocol.OpenAIResponsesV1), ctx, 0, false)
	if reply.GetImmediateResponse() != nil {
		t.Fatal("the initial text answer failed")
	}
	router.persistResponseObject(ctx)
	stored, err := store.GetResponse(t.Context(), ctx.SemanticResponse.ID)
	if err != nil {
		t.Fatal(err)
	}
	retained, _ := json.Marshal(stored.Input)
	t.Logf("stored input=%s", retained)
	if !bytes.Contains(retained, []byte(`"result":"`+streamedImage+`"`)) {
		t.Error("storage dropped the generated image from a valid client input")
	}
	history, err := materializeStoredResponseHistory(protocolcodec.NewBuiltinEngine(), []*responseapi.StoredResponse{stored})
	t.Logf("replayed input history: %v", err)
	if err != nil {
		t.Errorf("stored valid image input cannot be replayed: %v", err)
	} else {
		assertGeneratedImageHistoryResult(t, history)
	}
}

func assertGeneratedImageHistoryResult(t *testing.T, request *llmprotocol.Request) {
	t.Helper()
	count := 0
	for _, message := range request.Messages {
		for _, content := range message.Content {
			if image := content.GeneratedImage; image != nil {
				count++
				if image.Status != llmprotocol.ImageGenerationCompleted || image.Result == nil || *image.Result != streamedImage {
					t.Fatalf("replay changed the generated image: %+v", image)
				}
			}
		}
	}
	if count != 1 {
		t.Fatalf("replay retained %d generated images, want 1", count)
	}
}

// Counterfactual wire controls isolate the two lost fields without changing
// the production structs, serializer or validation policy.
func TestGeneratedImageHistoryRequiredFields(t *testing.T) {
	for _, tc := range []struct {
		name, fields string
		valid        bool
	}{
		{"neither", "", false},
		{"result_only", `,"result":"` + streamedImage + `"`, false},
		{"status_only", `,"status":"completed"`, false},
		{"both", `,"status":"completed","result":"` + streamedImage + `"`, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			item := json.RawMessage(`{"type":"image_generation_call","id":"ig_1"` + tc.fields + `}`)
			body, err := json.Marshal(map[string]interface{}{
				"model": "image-model", "store": false,
				"input": []json.RawMessage{item, json.RawMessage(`{"role":"user","content":"Make it blue"}`)},
			})
			if err != nil {
				t.Fatal(err)
			}
			_, _, _, err = protocolcodec.NewBuiltinEngine().DecodeRequest(llmprotocol.OpenAIResponsesV1, body)
			t.Logf("valid=%v decoder=%v", tc.valid, err)
			if (err == nil) != tc.valid {
				t.Fatalf("wire control disagrees: %v", err)
			}
		})
	}
}

func TestCloneGeneratedImageInputOwnsResult(t *testing.T) {
	original := []responseapi.InputItem{{Type: responseapi.ItemTypeImageGenerationCall, Result: json.RawMessage(`"aW1hZ2U="`)}}
	cloned := cloneResponseInputItems(original)
	cloned[0].Result[1] = 'X'
	if string(original[0].Result) != `"aW1hZ2U="` {
		t.Fatal("cloning an input snapshot retained the caller's result byte slice")
	}
}

func TestStoredGeneratedImageKeepsInvalidRecordsInvalid(t *testing.T) {
	for _, tc := range []struct{ fields, code string }{
		{"", "invalid_input_item_variant"},
		{`,"result":123`, "invalid_json"},
	} {
		var item responseapi.OutputItem
		if err := json.Unmarshal([]byte(`{"type":"image_generation_call","id":"ig_1","status":"completed"`+tc.fields+`}`), &item); err != nil {
			t.Fatal(err)
		}
		_, err := materializeStoredResponseHistory(protocolcodec.NewBuiltinEngine(), []*responseapi.StoredResponse{{Output: []responseapi.OutputItem{item}}})
		var protocolError *llmprotocol.ProtocolError
		if !errors.As(err, &protocolError) || protocolError.Category != llmprotocol.ErrorInvalidRequest || protocolError.Code != tc.code {
			t.Fatalf("a missing or wrongly typed result was accepted: %v", err)
		}
	}
}

func TestStoredGeneratedImageStatusIsVariantSpecific(t *testing.T) {
	inputs := []responseapi.InputItem{
		{Type: responseapi.ItemTypeImageGenerationCall, Status: "completed", Result: json.RawMessage(`null`)},
		{Type: responseapi.ItemTypeFunctionCall, Status: "completed"},
	}
	outputs := []responseapi.OutputItem{
		{Type: responseapi.ItemTypeImageGenerationCall, Status: "failed", Result: json.RawMessage(`null`)},
		{Type: responseapi.ItemTypeMessage, Status: "completed"},
	}
	for _, direction := range []string{"input", "output"} {
		t.Run(direction, func(t *testing.T) {
			items, err := marshalStoredInputItems(inputs)
			want := "completed"
			if direction == "output" {
				items, err = marshalStoredOutputItems(outputs)
				want = "failed"
			}
			if err != nil {
				t.Fatal(err)
			}
			var image, other map[string]json.RawMessage
			if err := json.Unmarshal(items[0], &image); err != nil {
				t.Fatal(err)
			}
			if err := json.Unmarshal(items[1], &other); err != nil {
				t.Fatal(err)
			}
			if string(image["status"]) != `"`+want+`"` || string(image["result"]) != "null" || other["status"] != nil {
				t.Fatalf("history shapes changed: image=%s other=%s", items[0], items[1])
			}
		})
	}
	if inputs[0].Status != "completed" || inputs[1].Status != "completed" || outputs[0].Status != "failed" || outputs[1].Status != "completed" {
		t.Fatal("history serialization mutated stored records")
	}
}
