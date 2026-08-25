package agentruntime

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"mime"
	"net/http"
	"net/url"
	"strings"
	"time"
	"unicode"
	"unicode/utf8"

	"github.com/google/uuid"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
)

const maximumPublicInferenceMetadataBytes = 256

// PublicInferenceClient deliberately accepts only a delegated public
// credential. Implementations must enter through the ordinary inference API;
// a worker may never dispatch directly to a backend or bypass access, quota,
// usage, and request-log settlement.
type PublicInferenceClient interface {
	Generate(
		context.Context,
		[]byte,
		llmprotocol.Request,
		func(llmprotocol.Event) error,
	) (PublicInferenceObservation, error)
}

// PublicInferenceObservation contains only Router-owned metadata from the
// public inference boundary. Token usage remains a semantic stream event so
// the model-step collector can retain only authoritative accounting.
type PublicInferenceObservation struct {
	RequestID           string
	SelectedRecipe      string
	SelectedDecision    string
	SelectedModel       string
	SelectedAlgorithm   string
	ResponsePath        string
	LatencyMilliseconds int64
	TTFTMilliseconds    *int64
}

type HTTPPublicInferenceOptions struct {
	Endpoint string
	Client   *http.Client
	Codecs   *protocolcodec.Registry
	Timeout  time.Duration
}

type HTTPPublicInferenceClient struct {
	endpoint string
	client   *http.Client
	engine   *protocolcodec.Engine
	timeout  time.Duration
	now      func() time.Time
}

// NewHTTPPublicInferenceClient binds the Agent worker to the deployment's
// public Envoy listener. The delegated credential therefore receives exactly
// the same policy, quota, routing, accounting, and logging behavior as any
// other API client. Endpoint is operator-owned process configuration, never a
// request or Tool Source value.
func NewHTTPPublicInferenceClient(options HTTPPublicInferenceOptions) (*HTTPPublicInferenceClient, error) {
	endpoint, err := normalizePublicInferenceEndpoint(options.Endpoint)
	if err != nil {
		return nil, err
	}
	if options.Codecs == nil {
		return nil, errors.New("agent public inference codecs are required")
	}
	engine, err := protocolcodec.NewEngine(options.Codecs, llmprotocol.DefaultPolicy())
	if err != nil {
		return nil, fmt.Errorf("compose Agent public inference codec: %w", err)
	}
	timeout := options.Timeout
	if timeout == 0 {
		timeout = 30 * time.Minute
	}
	if timeout < time.Second || timeout > 24*time.Hour {
		return nil, errors.New("agent public inference timeout is invalid")
	}
	client := http.Client{Transport: http.DefaultTransport}
	if options.Client != nil {
		client = *options.Client
	}
	client.CheckRedirect = func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }
	return &HTTPPublicInferenceClient{
		endpoint: endpoint,
		client:   &client,
		engine:   engine,
		timeout:  timeout,
		now:      time.Now,
	}, nil
}

func (client *HTTPPublicInferenceClient) Generate(
	ctx context.Context,
	credential []byte,
	request llmprotocol.Request,
	emit func(llmprotocol.Event) error,
) (PublicInferenceObservation, error) {
	if client == nil || client.client == nil || client.engine == nil || len(credential) == 0 || emit == nil {
		return PublicInferenceObservation{}, errors.New("agent public inference client is unavailable")
	}
	now := client.now
	if now == nil {
		now = time.Now
	}
	callContext, cancel := context.WithTimeout(ctx, client.timeout)
	defer cancel()
	startedAt := now()
	requestID := uuid.NewString()
	request.Stream = true
	request.Generation++
	encoded, err := client.engine.EncodeRequest(llmprotocol.OpenAIChatV1, request, llmprotocol.Envelope{})
	if err != nil {
		return PublicInferenceObservation{}, fmt.Errorf("encode Agent inference request: %w", err)
	}
	defer clear(encoded.Body)

	httpRequest, err := http.NewRequestWithContext(
		callContext, http.MethodPost, client.endpoint, bytes.NewReader(encoded.Body),
	)
	if err != nil {
		return PublicInferenceObservation{}, errors.New("create Agent public inference request")
	}
	httpRequest.Header.Set("Authorization", "Bearer "+string(credential))
	httpRequest.Header.Set("Content-Type", "application/json")
	httpRequest.Header.Set("Accept", "text/event-stream")
	httpRequest.Header.Set("Cache-Control", "no-store")
	httpRequest.Header.Set(headers.RequestID, requestID)

	response, err := client.client.Do(httpRequest)
	if err != nil {
		return PublicInferenceObservation{}, fmt.Errorf("agent public inference request failed: %w", err)
	}
	defer response.Body.Close()
	if response.StatusCode < http.StatusOK || response.StatusCode >= http.StatusMultipleChoices {
		_, _ = io.CopyN(io.Discard, response.Body, 64<<10)
		return PublicInferenceObservation{}, fmt.Errorf("agent public inference returned HTTP %d", response.StatusCode)
	}
	mediaType, _, err := mime.ParseMediaType(response.Header.Get("Content-Type"))
	if err != nil || mediaType != "text/event-stream" {
		_, _ = io.CopyN(io.Discard, response.Body, 64<<10)
		return PublicInferenceObservation{}, errors.New("agent public inference did not return a semantic event stream")
	}
	observation := publicInferenceObservationFromHeaders(response.Header, requestID)
	stream, err := client.engine.NewStream(
		llmprotocol.OpenAIChatV1,
		llmprotocol.OpenAIChatV1,
		llmprotocol.StreamContext{
			Context:     callContext,
			PublicModel: request.Model,
		},
	)
	if err != nil {
		return PublicInferenceObservation{}, fmt.Errorf("open Agent inference stream: %w", err)
	}
	var firstOutputAt *time.Time
	emitObserved := func(event llmprotocol.Event) error {
		if firstOutputAt == nil && publicInferenceOutputStarted(event) {
			seenAt := now()
			firstOutputAt = &seenAt
		}
		return emit(event)
	}
	buffer := make([]byte, 32<<10)
	for {
		count, readErr := response.Body.Read(buffer)
		if count > 0 {
			_, events, _, decodeErr := stream.Push(buffer[:count])
			for _, event := range events {
				if emitErr := emitObserved(event); emitErr != nil {
					return PublicInferenceObservation{}, emitErr
				}
			}
			if decodeErr != nil {
				return PublicInferenceObservation{}, fmt.Errorf("decode Agent inference stream: %w", decodeErr)
			}
		}
		if readErr != nil {
			if !errors.Is(readErr, io.EOF) {
				_, events, _, _ := stream.Finalize(readErr)
				for _, event := range events {
					_ = emitObserved(event)
				}
				return PublicInferenceObservation{}, fmt.Errorf("read Agent inference stream: %w", readErr)
			}
			break
		}
	}
	_, events, _, err := stream.Finalize(nil)
	for _, event := range events {
		if emitErr := emitObserved(event); emitErr != nil {
			return PublicInferenceObservation{}, emitErr
		}
	}
	if err != nil {
		return PublicInferenceObservation{}, err
	}
	completedAt := now()
	observation.LatencyMilliseconds = nonNegativeMilliseconds(completedAt.Sub(startedAt))
	if firstOutputAt != nil {
		ttft := nonNegativeMilliseconds(firstOutputAt.Sub(startedAt))
		if ttft > observation.LatencyMilliseconds {
			ttft = observation.LatencyMilliseconds
		}
		observation.TTFTMilliseconds = &ttft
	}
	return observation, nil
}

func publicInferenceObservationFromHeaders(
	values http.Header, requestID string,
) PublicInferenceObservation {
	return PublicInferenceObservation{
		RequestID:         requestID,
		SelectedRecipe:    publicInferenceMetadataValue(values.Get(headers.VSRSelectedRecipe)),
		SelectedDecision:  publicInferenceMetadataValue(values.Get(headers.VSRSelectedDecision)),
		SelectedModel:     publicInferenceMetadataValue(values.Get(headers.VSRSelectedModel)),
		SelectedAlgorithm: publicInferenceMetadataValue(values.Get(headers.VSRSelectedAlgorithm)),
		ResponsePath:      publicInferenceResponsePath(values.Get(headers.VSRResponsePath)),
	}
}

func publicInferenceMetadataValue(value string) string {
	value = strings.TrimSpace(value)
	if value == "" || len(value) > maximumPublicInferenceMetadataBytes || !utf8.ValidString(value) {
		return ""
	}
	for _, character := range value {
		if unicode.IsControl(character) || !unicode.IsGraphic(character) {
			return ""
		}
	}
	return value
}

func publicInferenceResponsePath(value string) string {
	value = publicInferenceMetadataValue(value)
	switch value {
	case headers.ResponsePathUpstream, headers.ResponsePathCache, headers.ResponsePathFastResponse,
		headers.ResponsePathLooper, headers.ResponsePathImageGeneration:
		return value
	default:
		return ""
	}
}

func publicInferenceOutputStarted(event llmprotocol.Event) bool {
	switch event.Type {
	case llmprotocol.EventOutputTextDelta, llmprotocol.EventReasoningDelta:
		return event.Delta != ""
	case llmprotocol.EventToolCallDelta:
		return event.ToolCall != nil
	case llmprotocol.EventOutputItemStarted:
		return event.ToolCall != nil || event.Content != nil
	default:
		return false
	}
}

func nonNegativeMilliseconds(value time.Duration) int64 {
	if value <= 0 {
		return 0
	}
	return value.Milliseconds()
}

func normalizePublicInferenceEndpoint(raw string) (string, error) {
	parsed, err := url.Parse(strings.TrimSpace(raw))
	if err != nil || (parsed.Scheme != "http" && parsed.Scheme != "https") || parsed.Host == "" ||
		parsed.User != nil || parsed.RawQuery != "" || parsed.Fragment != "" {
		return "", errors.New("agent public inference endpoint is invalid")
	}
	path := strings.TrimSuffix(parsed.EscapedPath(), "/")
	switch path {
	case "":
		parsed.Path = "/v1/chat/completions"
	case "/v1":
		parsed.Path = strings.TrimSuffix(parsed.Path, "/") + "/chat/completions"
	case "/v1/chat/completions":
	default:
		return "", errors.New("agent public inference endpoint must address /v1/chat/completions")
	}
	parsed.RawPath = ""
	return parsed.String(), nil
}

var _ PublicInferenceClient = (*HTTPPublicInferenceClient)(nil)
