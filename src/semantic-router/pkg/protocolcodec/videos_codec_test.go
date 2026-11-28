package protocolcodec

import (
	"encoding/json"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func TestVideosCodecCreateRoundTrip(t *testing.T) {
	codec := VideosCodec{}
	request, _, _, err := codec.DecodeRequest([]byte(`{"model":"video-route","prompt":"a lighthouse","seconds":"8","size":"1280x720","width":1280,"height":720,"fps":24,"num_frames":192,"quality":"high"}`), llmprotocol.Policy{})
	if err != nil {
		t.Fatal(err)
	}
	options := request.VideoGeneration
	if request.Model != "video-route" || options == nil || options.Prompt != "a lighthouse" ||
		options.Seconds != "8" || options.Size != "1280x720" || options.Width != 1280 || options.Height != 720 ||
		options.FPS != 24 || options.NumFrames != 192 || options.Quality != "high" {
		t.Fatalf("decoded request = %+v", request)
	}
	body, _, err := codec.EncodeRequest(request, llmprotocol.Envelope{}, llmprotocol.Policy{})
	if err != nil {
		t.Fatal(err)
	}
	want := `{"model":"video-route","prompt":"a lighthouse","seconds":"8","size":"1280x720","width":1280,"height":720,"fps":24,"num_frames":192,"quality":"high"}`
	if string(body) != want {
		t.Fatalf("encoded request = %s, want %s", body, want)
	}
}

func TestVideosCodecRejectsModelLessCreate(t *testing.T) {
	_, _, _, err := (VideosCodec{}).DecodeRequest([]byte(`{"prompt":"a lighthouse"}`), llmprotocol.Policy{})
	var protocolErr *llmprotocol.ProtocolError
	if err == nil || !strings.Contains(err.Error(), "invalid_video_request") {
		t.Fatalf("model-less create error = %v", err)
	}
	if !asProtocolError(err, &protocolErr) || protocolErr.Code != "invalid_video_request" {
		t.Fatalf("model-less create error = %+v", protocolErr)
	}
}

func TestVideosCodecJobResponseRoundTrip(t *testing.T) {
	codec := VideosCodec{}
	body := []byte(`{"id":"video_1","object":"video","status":"in_progress","progress":42,"seconds":"8","size":"1280x720","created_at":1701234567,"expires_at":1701238167}`)
	response, _, _, err := codec.DecodeResponse(body, llmprotocol.Policy{})
	if err != nil {
		t.Fatal(err)
	}
	if len(response.Output) != 1 || len(response.Output[0].Content) != 1 {
		t.Fatalf("output = %+v", response.Output)
	}
	video := response.Output[0].Content[0].GeneratedVideo
	if video == nil || video.ID != "video_1" || video.Status != llmprotocol.VideoGenerationInProgress ||
		video.Progress == nil || *video.Progress != 42 || video.CreatedAt == nil || video.CreatedAt.Unix() != 1701234567 {
		t.Fatalf("decoded video = %+v", video)
	}
	encoded, _, err := codec.EncodeResponse(response, llmprotocol.Envelope{}, llmprotocol.Policy{})
	if err != nil {
		t.Fatal(err)
	}
	var wire map[string]any
	if err := json.Unmarshal(encoded, &wire); err != nil {
		t.Fatal(err)
	}
	if wire["object"] != "video" || wire["id"] != "video_1" || wire["progress"] != float64(42) {
		t.Fatalf("encoded job = %s", encoded)
	}
}

func TestVideosCodecDeleteRoundTrip(t *testing.T) {
	codec := VideosCodec{}
	response, _, _, err := codec.DecodeResponse([]byte(`{"id":"video_1","deleted":true,"object":"video.deleted"}`), llmprotocol.Policy{})
	if err != nil {
		t.Fatal(err)
	}
	video := response.Output[0].Content[0].GeneratedVideo
	if !video.Deleted || video.ID != "video_1" || video.Status != llmprotocol.VideoGenerationCompleted {
		t.Fatalf("decoded delete = %+v", video)
	}
	body, _, err := codec.EncodeResponse(response, llmprotocol.Envelope{}, llmprotocol.Policy{})
	if err != nil {
		t.Fatal(err)
	}
	var wire map[string]any
	if err := json.Unmarshal(body, &wire); err != nil {
		t.Fatal(err)
	}
	if wire["object"] != "video.deleted" || wire["deleted"] != true || wire["id"] != "video_1" {
		t.Fatalf("encoded delete = %s", body)
	}
	if _, found := wire["status"]; found {
		t.Fatalf("delete object invented a job status: %s", body)
	}
}

func TestVideosCodecRejectsUnknownStatus(t *testing.T) {
	if _, _, _, err := (VideosCodec{}).DecodeResponse([]byte(`{"id":"video_1","status":"pending"}`), llmprotocol.Policy{}); err == nil {
		t.Fatal("unknown video status must fail closed")
	}
}

func TestVideosCodecErrorBodyIsTransportError(t *testing.T) {
	codec := VideosCodec{}
	if _, _, _, err := codec.DecodeResponse([]byte(`{"error":{"message":"bad prompt","code":"invalid_prompt"}}`), llmprotocol.Policy{}); err == nil {
		t.Fatal("an error body must not decode as a video job")
	}
	transportError, _, err := codec.DecodeTransportError([]byte(`{"error":{"message":"bad prompt","code":"invalid_prompt"}}`), llmprotocol.Policy{})
	if err != nil {
		t.Fatal(err)
	}
	if transportError.Error == nil || transportError.Error.Code != "invalid_prompt" || transportError.Error.Message != "bad prompt" {
		t.Fatalf("transport error = %+v", transportError.Error)
	}
}

func TestVideosCodecCapabilities(t *testing.T) {
	capabilities := (VideosCodec{}).Capabilities()
	if !capabilities.Supports(llmprotocol.CapabilityText) || !capabilities.Supports(llmprotocol.CapabilityVideoGeneration) {
		t.Fatalf("videos capabilities = %v", capabilities.Names())
	}
}

func TestVideosCodecDeletedResponseOmitsStatus(t *testing.T) {
	body, _, err := (VideosCodec{}).EncodeResponse(llmprotocol.Response{Output: []llmprotocol.OutputItem{{Content: []llmprotocol.Content{{
		Kind: llmprotocol.ContentGeneratedVideo, GeneratedVideo: &llmprotocol.GeneratedVideo{
			ID: "video_1", Status: llmprotocol.VideoGenerationCompleted, Deleted: true,
			CreatedAt: llmprotocolTime(time.Unix(1, 0)),
		},
	}}}}}, llmprotocol.Envelope{}, llmprotocol.Policy{})
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(body), `"status"`) {
		t.Fatalf("delete response contains status: %s", body)
	}
}

func asProtocolError(err error, target **llmprotocol.ProtocolError) bool {
	for err != nil {
		var protocolErr *llmprotocol.ProtocolError
		if errors.As(err, &protocolErr) {
			*target = protocolErr
			return true
		}
		unwrapper, ok := err.(interface{ Unwrap() error })
		if !ok {
			return false
		}
		err = unwrapper.Unwrap()
	}
	return false
}

func llmprotocolTime(value time.Time) *time.Time { return &value }

func TestVideosRequestKeepsFractionalFPSAndTheRewrittenPrompt(t *testing.T) {
	codec := VideosCodec{}
	request, _, _, err := codec.DecodeRequest([]byte(`{"model":"video-route","prompt":"a cat","fps":23.976}`), llmprotocol.Policy{})
	if err != nil {
		t.Fatal(err)
	}
	if request.VideoGeneration == nil || request.VideoGeneration.FPS != 23.976 {
		t.Fatalf("fps = %+v", request.VideoGeneration)
	}
	request.Messages[0].Content = []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "a dog"}}
	wire, _, err := codec.EncodeRequest(request, llmprotocol.Envelope{}, llmprotocol.Policy{})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(wire), `"prompt":"a dog"`) || !strings.Contains(string(wire), `"fps":23.976`) {
		t.Fatalf("wire = %s", wire)
	}
}
