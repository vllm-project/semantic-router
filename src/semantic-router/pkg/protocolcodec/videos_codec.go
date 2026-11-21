package protocolcodec

import (
	"bytes"
	"encoding/json"
	"fmt"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// VideosCodec is the OpenAI Videos job dialect. It retains only the request,
// job, and delete fields modelled below; provider-only response fields are
// dropped exactly as ImagesCodec drops revised_prompt and url.
type VideosCodec struct{}

func (VideosCodec) Format() llmprotocol.WireFormat { return llmprotocol.OpenAIVideosV1 }
func (VideosCodec) Stateless() bool                { return true }
func (VideosCodec) Capabilities() llmprotocol.CapabilitySet {
	return llmprotocol.Capabilities(llmprotocol.CapabilityText, llmprotocol.CapabilityVideoGeneration)
}

type videosRequestWire struct {
	Model     string  `json:"model,omitempty"`
	Prompt    string  `json:"prompt,omitempty"`
	Seconds   string  `json:"seconds,omitempty"`
	Size      string  `json:"size,omitempty"`
	Width     int64   `json:"width,omitempty"`
	Height    int64   `json:"height,omitempty"`
	FPS       float64 `json:"fps,omitempty"`
	NumFrames int64   `json:"num_frames,omitempty"`
	Quality   string  `json:"quality,omitempty"`
}

type videosResponseWire struct {
	ID          string           `json:"id,omitempty"`
	Object      string           `json:"object,omitempty"`
	Status      string           `json:"status,omitempty"`
	Progress    *int64           `json:"progress,omitempty"`
	Seconds     string           `json:"seconds,omitempty"`
	Size        string           `json:"size,omitempty"`
	CreatedAt   *int64           `json:"created_at,omitempty"`
	CompletedAt *int64           `json:"completed_at,omitempty"`
	ExpiresAt   *int64           `json:"expires_at,omitempty"`
	Deleted     bool             `json:"deleted,omitempty"`
	Error       *videosErrorWire `json:"error,omitempty"`
}

type videosErrorWire struct {
	Message string `json:"message"`
}

func (VideosCodec) DecodeRequest(body []byte, _ llmprotocol.Policy) (llmprotocol.Request, llmprotocol.Envelope, llmprotocol.Diagnostics, error) {
	var wire videosRequestWire
	if err := json.Unmarshal(body, &wire); err != nil {
		return llmprotocol.Request{}, llmprotocol.Envelope{}, nil, invalidVideoRequest(err)
	}
	if wire.Model == "" {
		return llmprotocol.Request{}, llmprotocol.Envelope{}, nil, invalidVideoRequest(fmt.Errorf("videos model is required"))
	}
	request := llmprotocol.Request{
		Generation: 1,
		Model:      wire.Model,
		VideoGeneration: &llmprotocol.VideoGenerationOptions{
			Prompt: wire.Prompt, Model: wire.Model, Seconds: wire.Seconds, Size: wire.Size,
			Width: wire.Width, Height: wire.Height, FPS: wire.FPS, NumFrames: wire.NumFrames,
			Quality: wire.Quality,
		},
	}
	if wire.Prompt != "" {
		request.Messages = []llmprotocol.Message{{
			Role:    llmprotocol.RoleUser,
			Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: wire.Prompt}},
		}}
	}
	return request, llmprotocol.Envelope{}, nil, nil
}

func (VideosCodec) EncodeRequest(request llmprotocol.Request, _ llmprotocol.Envelope, _ llmprotocol.Policy) ([]byte, llmprotocol.Diagnostics, error) {
	options := request.VideoGeneration
	if options == nil {
		return nil, nil, unsupportedDownstreamTranslation(fmt.Errorf("videos wire requires a video-generation request"))
	}
	prompt := options.Prompt
	if text, ok := lastUserRequestText(request); ok && text != "" {
		// The routing projection of the prompt is authoritative: a plugin that
		// rewrites the user turn must reach the backend, not the original.
		prompt = text
	}
	wire := videosRequestWire{
		Model: request.Model, Prompt: prompt, Seconds: options.Seconds, Size: options.Size,
		Width: options.Width, Height: options.Height, FPS: options.FPS, NumFrames: options.NumFrames,
		Quality: options.Quality,
	}
	body, err := json.Marshal(wire)
	if err != nil {
		return nil, nil, fmt.Errorf("encode videos request: %w", err)
	}
	return body, nil, nil
}

func (VideosCodec) DecodeResponse(body []byte, _ llmprotocol.Policy) (llmprotocol.Response, llmprotocol.Envelope, llmprotocol.Diagnostics, error) {
	var wire videosResponseWire
	if err := json.Unmarshal(body, &wire); err != nil {
		return llmprotocol.Response{}, llmprotocol.Envelope{}, nil, invalidUpstreamResponse(err)
	}
	if wire.Error != nil && wire.ID == "" && wire.Status == "" && !wire.Deleted {
		return llmprotocol.Response{}, llmprotocol.Envelope{}, nil, invalidUpstreamResponse(fmt.Errorf("videos upstream error: %s", wire.Error.Message))
	}
	status := llmprotocol.VideoGenerationStatus(wire.Status)
	if wire.Deleted {
		status = llmprotocol.VideoGenerationCompleted
	} else if !llmprotocol.ValidVideoGenerationStatus(status) {
		return llmprotocol.Response{}, llmprotocol.Envelope{}, nil, invalidUpstreamResponse(fmt.Errorf("videos response status %q is invalid", wire.Status))
	}
	video := &llmprotocol.GeneratedVideo{
		ID: wire.ID, Status: status, Progress: wire.Progress, Seconds: wire.Seconds, Size: wire.Size,
		CreatedAt: videosTime(wire.CreatedAt), CompletedAt: videosTime(wire.CompletedAt), ExpiresAt: videosTime(wire.ExpiresAt),
		Deleted: wire.Deleted,
	}
	if wire.Error != nil {
		video.Err = llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "video_generation_failed", wire.Error.Message, nil)
	}
	response := llmprotocol.Response{
		Generation: 1,
		ID:         llmprotocol.StableID("videos-response", wire.ID),
		Usage:      llmprotocol.Usage{State: llmprotocol.UsageUnavailable},
		Output: []llmprotocol.OutputItem{{
			ID:   llmprotocol.StableID("videos-output", wire.ID),
			Role: llmprotocol.RoleAssistant,
			Content: []llmprotocol.Content{{
				Kind:           llmprotocol.ContentGeneratedVideo,
				GeneratedVideo: video,
			}},
		}},
	}
	if video.CreatedAt != nil {
		response.CreatedAt = *video.CreatedAt
	}
	return response, llmprotocol.Envelope{}, nil, nil
}

func (VideosCodec) EncodeResponse(response llmprotocol.Response, _ llmprotocol.Envelope, _ llmprotocol.Policy) ([]byte, llmprotocol.Diagnostics, error) {
	var video *llmprotocol.GeneratedVideo
	for _, item := range response.Output {
		for _, content := range item.Content {
			if content.Kind == llmprotocol.ContentGeneratedVideo && content.GeneratedVideo != nil {
				video = content.GeneratedVideo
				break
			}
		}
		if video != nil {
			break
		}
	}
	if video == nil {
		return nil, nil, unsupportedDownstreamTranslation(fmt.Errorf("videos response contains no generated video"))
	}
	// A delete answers with the removed id and nothing else; a job object keeps
	// its lifecycle fields.
	wire := videosResponseWire{ID: video.ID, Object: "video", Deleted: video.Deleted}
	if !video.Deleted {
		wire.Status = string(video.Status)
		wire.Progress, wire.Seconds, wire.Size = video.Progress, video.Seconds, video.Size
		wire.CreatedAt, wire.CompletedAt, wire.ExpiresAt = videosUnix(video.CreatedAt), videosUnix(video.CompletedAt), videosUnix(video.ExpiresAt)
		if video.Err != nil {
			wire.Error = &videosErrorWire{Message: video.Err.Message}
		}
	} else {
		wire.Object = "video.deleted"
	}
	body, err := json.Marshal(wire)
	if err != nil {
		return nil, nil, fmt.Errorf("encode videos response: %w", err)
	}
	return body, nil, nil
}

func (VideosCodec) DecodeTransportError(body []byte, _ llmprotocol.Policy) (llmprotocol.TransportError, llmprotocol.Diagnostics, error) {
	var wire struct {
		Error *struct {
			Message string          `json:"message"`
			Code    openAIErrorCode `json:"code"`
		} `json:"error"`
	}
	if len(bytes.TrimSpace(body)) > 0 {
		if err := json.Unmarshal(body, &wire); err != nil {
			return llmprotocol.TransportError{}, nil, invalidUpstreamResponse(err)
		}
	}
	message := "video service returned an error"
	code := "invalid_upstream_error"
	if wire.Error != nil {
		if wire.Error.Message != "" {
			message = wire.Error.Message
		}
		if wire.Error.Code != "" {
			code = string(wire.Error.Code)
		}
	}
	return llmprotocol.TransportError{Error: llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, code, message, nil)}, nil, nil
}

func (VideosCodec) EncodeTransportError(err llmprotocol.TransportError) []byte {
	message := "model service returned an invalid error response"
	code := "invalid_upstream_error"
	if err.Error != nil {
		if err.Error.Message != "" {
			message = err.Error.Message
		}
		if err.Error.Code != "" {
			code = err.Error.Code
		}
	}
	body, marshalErr := json.Marshal(map[string]any{
		"error": map[string]any{"message": message, "type": "invalid_request_error", "code": code},
	})
	if marshalErr != nil {
		return []byte(`{"error":{"message":"model service returned an invalid error response","type":"invalid_request_error","code":"invalid_upstream_error"}}`)
	}
	return body
}

func (VideosCodec) NewDecoder(_ llmprotocol.StreamContext, _ llmprotocol.Policy) llmprotocol.StreamDecoder {
	return videosStreamUnsupportedDecoder{}
}

func (VideosCodec) NewEncoder(_ llmprotocol.StreamContext, _ llmprotocol.Policy) llmprotocol.StreamEncoder {
	return videosStreamUnsupportedEncoder{}
}

type (
	videosStreamUnsupportedDecoder struct{}
	videosStreamUnsupportedEncoder struct{}
)

func (videosStreamUnsupportedDecoder) Push([]byte) ([]llmprotocol.Event, llmprotocol.Diagnostics, error) {
	return nil, nil, fmt.Errorf("videos wire does not support streaming")
}

func (videosStreamUnsupportedDecoder) Finalize(error) ([]llmprotocol.Event, llmprotocol.Diagnostics, error) {
	return nil, nil, fmt.Errorf("videos wire does not support streaming")
}

func (videosStreamUnsupportedEncoder) Push(llmprotocol.Event) ([][]byte, llmprotocol.Diagnostics, error) {
	return nil, nil, fmt.Errorf("videos wire does not support streaming")
}

func (videosStreamUnsupportedEncoder) Finalize(error) ([][]byte, llmprotocol.Diagnostics, error) {
	return nil, nil, fmt.Errorf("videos wire does not support streaming")
}

func invalidVideoRequest(cause error) error {
	return llmprotocol.NewError(llmprotocol.ErrorInvalidRequest, "invalid_video_request", cause.Error(), cause)
}

func videosTime(value *int64) *time.Time {
	if value == nil {
		return nil
	}
	parsed := time.Unix(*value, 0)
	return &parsed
}

func videosUnix(value *time.Time) *int64 {
	if value == nil {
		return nil
	}
	unix := value.Unix()
	return &unix
}
