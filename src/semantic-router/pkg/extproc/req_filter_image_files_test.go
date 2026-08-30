package extproc

import (
	"encoding/base64"
	"encoding/json"
	"errors"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/responseapi"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/vectorstore"
)

var imageFilePNGBytes = []byte("\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR")

func newImageFileRouter(t *testing.T) (*OpenAIRouter, *vectorstore.FileStore) {
	t.Helper()
	fileStore, err := vectorstore.NewFileStore(t.TempDir(), vectorstore.NewMemoryMetadataRegistry())
	if err != nil {
		t.Fatalf("NewFileStore(): %v", err)
	}
	registry := routerruntime.NewRegistry(nil)
	registry.SetVectorStoreRuntime(&routerruntime.VectorStoreRuntime{FileStore: fileStore})
	return &OpenAIRouter{RuntimeRegistry: registry}, fileStore
}

func saveImageFile(t *testing.T, fileStore *vectorstore.FileStore, filename string, content []byte) string {
	t.Helper()
	record, err := fileStore.Save(filename, content, "vision")
	if err != nil {
		t.Fatalf("Save(%s): %v", filename, err)
	}
	return record.ID
}

func responsesImageFileRequest(fileID string) []byte {
	return []byte(`{"model":"vision-model","input":[{"type":"message","role":"user","content":[` +
		`{"type":"input_text","text":"what is shown here?"},` +
		`{"type":"input_image","file_id":"` + fileID + `","detail":"high"}]}]}`)
}

func decodeImageFileRequest(t *testing.T, router *OpenAIRouter, body []byte) (*llmprotocol.Request, *RequestContext) {
	t.Helper()
	ctx := &RequestContext{
		SourceFormat: llmprotocol.OpenAIResponsesV1,
		TargetFormat: llmprotocol.OpenAIChatV1,
		RequestID:    "request_image_file",
		TraceContext: t.Context(),
	}
	request, immediate := router.prepareProtocolRequest(body, ctx)
	if immediate != nil || request == nil {
		t.Fatalf("request was rejected: request=%+v immediate=%+v", request, immediate)
	}
	return request, ctx
}

func firstImageContent(t *testing.T, request *llmprotocol.Request) llmprotocol.Content {
	t.Helper()
	for _, message := range request.Messages {
		for _, content := range message.Content {
			if content.Kind == llmprotocol.ContentImage {
				return content
			}
		}
	}
	t.Fatal("request has no image content")
	return llmprotocol.Content{}
}

// A file_id image must count as image input for routing signals even though
// no URL or inline data is present at ingress.
func TestResponsesFileIDImageCountsAsImageInput(t *testing.T) {
	request, _ := decodeImageFileRequest(t, &OpenAIRouter{}, responsesImageFileRequest("file_abc123"))
	snapshot := extractSemanticRequestSignals(request)
	if snapshot.ImageContentCount != 1 {
		t.Fatalf("ImageContentCount = %d, want 1", snapshot.ImageContentCount)
	}
	if snapshot.UserContent != "what is shown here?" {
		t.Fatalf("UserContent = %q", snapshot.UserContent)
	}
}

func TestResolveImageFileReferencesInlinesRouterOwnedFiles(t *testing.T) {
	router, fileStore := newImageFileRouter(t)
	fileID := saveImageFile(t, fileStore, "cat.png", imageFilePNGBytes)
	request, ctx := decodeImageFileRequest(t, router, responsesImageFileRequest(fileID))

	changed, err := router.resolveImageFileReferences(request)
	if err != nil {
		t.Fatalf("resolveImageFileReferences(): %v", err)
	}
	if !changed {
		t.Fatal("resolveImageFileReferences() reported no change")
	}
	image := firstImageContent(t, request)
	payload := base64.StdEncoding.EncodeToString(imageFilePNGBytes)
	if image.FileID != "" || image.MediaType != "image/png" || image.Data != payload || image.Detail != "high" {
		t.Fatalf("inlined image = %+v", image)
	}

	body, err := router.encodeDispatchRequest(ctx)
	if err != nil {
		t.Fatalf("encodeDispatchRequest(): %v", err)
	}
	if !strings.Contains(string(body), `"url":"data:image/png;base64,`+payload+`"`) {
		t.Fatalf("Chat request lacks the inlined image:\n%s", body)
	}
	if strings.Contains(string(body), fileID) {
		t.Fatalf("Chat request leaks the Router file ID:\n%s", body)
	}
}

func TestResolveImageFileReferencesLeavesForeignFilesUntouched(t *testing.T) {
	router, _ := newImageFileRouter(t)
	for name, candidate := range map[string]*OpenAIRouter{"unknown file": router, "no file store": {}} {
		t.Run(name, func(t *testing.T) {
			request, _ := decodeImageFileRequest(t, candidate, responsesImageFileRequest("file-provider-owned"))
			changed, err := candidate.resolveImageFileReferences(request)
			if err != nil || changed {
				t.Fatalf("resolveImageFileReferences() = %v, %v; want false, nil", changed, err)
			}
			if image := firstImageContent(t, request); image.FileID != "file-provider-owned" || image.Data != "" {
				t.Fatalf("foreign reference was altered: %+v", image)
			}
		})
	}
}

func TestResolveImageFileReferencesRejectsNonImageUploads(t *testing.T) {
	router, fileStore := newImageFileRouter(t)
	fileID := saveImageFile(t, fileStore, "notes.png", []byte("just some text pretending to be an image"))
	request, _ := decodeImageFileRequest(t, router, responsesImageFileRequest(fileID))

	_, err := router.resolveImageFileReferences(request)
	var protocolError *llmprotocol.ProtocolError
	if !errors.As(err, &protocolError) || protocolError.Code != "image_file_unsupported" || protocolError.Category != llmprotocol.ErrorInvalidRequest {
		t.Fatalf("resolveImageFileReferences() error = %v, want image_file_unsupported", err)
	}
	if !strings.Contains(err.Error(), fileID) {
		t.Fatalf("error does not name the file: %v", err)
	}
}

// Images referenced by file_id in retained previous_response_id history are
// materialized into the neutral request before dispatch and must be inlined
// on the same path as the current turn.
func TestPrepareProviderRequestInlinesRetainedHistoryImages(t *testing.T) {
	router, fileStore := newImageFileRouter(t)
	fileID := saveImageFile(t, fileStore, "cat.png", imageFilePNGBytes)
	request, ctx := decodeImageFileRequest(t, router, []byte(`{"model":"vision-model","input":"and now?"}`))
	ctx.ResponseObjectState = &ResponseObjectState{
		GeneratedResponseID: "resp_current",
		ConversationHistory: []*responseapi.StoredResponse{{
			ID: "resp_previous", Object: "response", Model: "vision-model", Status: responseapi.StatusCompleted,
			Input: []responseapi.InputItem{{
				ID: "item_previous", Type: responseapi.ItemTypeMessage, Role: responseapi.RoleUser,
				Content: json.RawMessage(`[{"type":"input_image","file_id":"` + fileID + `"}]`),
			}},
			OutputText: "A cat.",
		}},
	}

	changed, err := router.prepareProviderRequest(request, &providerDispatch{
		upstreamModel: "vision-model", targetFormat: llmprotocol.OpenAIChatV1,
	}, ctx)
	if err != nil {
		t.Fatalf("prepareProviderRequest(): %v", err)
	}
	if !changed {
		t.Fatal("prepareProviderRequest() reported no change")
	}
	image := firstImageContent(t, request)
	if image.FileID != "" || image.MediaType != "image/png" || image.Data != base64.StdEncoding.EncodeToString(imageFilePNGBytes) {
		t.Fatalf("history image was not inlined: %+v", image)
	}
	if request.Messages[0].Role != llmprotocol.RoleUser || request.Messages[len(request.Messages)-1].Role != llmprotocol.RoleUser {
		t.Fatalf("unexpected message order: %+v", request.Messages)
	}
}
