package backendinvoker

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"strings"
	"sync"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
)

var builtinCodecs = protocolcodec.NewBuiltinRegistry()

func (i *Invoker) codecRegistry() *protocolcodec.Registry {
	if i != nil && i.Codecs != nil {
		return i.Codecs
	}
	return builtinCodecs
}

func (i *Invoker) WireCapabilities() []protocolcodec.Capability {
	return i.codecRegistry().Capabilities()
}

func (i *Invoker) transformResponse(
	ctx context.Context,
	plan Plan,
	backend Backend,
	attempt AttemptResult,
	response *http.Response,
	sensitiveValues []string,
) (*http.Response, error) {
	if response == nil || response.Body == nil {
		return nil, fmt.Errorf("backend returned an empty response")
	}
	sourceBody := response.Body
	closeSourceBody := true
	defer func() {
		if closeSourceBody {
			_ = sourceBody.Close()
		}
	}()
	engine, transformResponseErr := protocolcodec.NewEngine(i.codecRegistry(), llmprotocol.DefaultPolicy())
	if transformResponseErr != nil {
		return nil, transformResponseErr
	}
	if response.StatusCode < http.StatusOK || response.StatusCode >= http.StatusMultipleChoices {
		body, readErr := readWireBody(sourceBody, maximumProviderErrorBodyBytes)
		closeSourceBody = false
		protocolError, publicBody, providerRequestID := translateUpstreamHTTPError(
			engine,
			backend.WireFormat,
			plan.SourceFormat,
			response.StatusCode,
			response.Header,
			body,
			readErr,
			sensitiveValues,
		)
		if err := i.finalizeResponse(ctx, plan, attempt, ResponseTerminal{
			Usage:      llmprotocol.Usage{State: llmprotocol.UsageUnavailable},
			StopReason: llmprotocol.StopError,
			Error:      protocolError,
		}); err != nil {
			return nil, err
		}
		response.Body = io.NopCloser(bytes.NewReader(publicBody))
		response.ContentLength = int64(len(publicBody))
		response.Header = publicUpstreamErrorHeaders(
			len(publicBody),
			protocolError,
			providerRequestID,
			sensitiveValues,
		)
		return response, nil
	}
	if plan.Streaming && response.StatusCode >= 200 && response.StatusCode < 300 &&
		strings.Contains(strings.ToLower(response.Header.Get("Content-Type")), "text/event-stream") {
		stream, err := engine.NewStreamWithMutation(
			backend.WireFormat,
			plan.SourceFormat,
			llmprotocol.StreamContext{
				Context: ctx, Source: backend.WireFormat, Target: plan.SourceFormat,
				PublicModel: plan.ModelID, ProviderModel: backend.ProviderModelID,
			},
			publicStreamErrorMutation(sensitiveValues),
		)
		if err != nil {
			return nil, err
		}
		response.Body = newCodecStreamBody(sourceBody, stream, func(terminal ResponseTerminal) error {
			return i.finalizeResponse(ctx, plan, attempt, terminal)
		})
		closeSourceBody = false
		response.ContentLength = -1
		response.Header.Del("Content-Length")
		response.Header.Set("Content-Type", "text/event-stream")
		return response, nil
	}
	body, transformResponseErr := readWireBody(sourceBody, llmprotocol.DefaultPolicy().Limits.BodyBytes)
	closeSourceBody = false
	if transformResponseErr != nil {
		return nil, transformResponseErr
	}
	translated, transformResponseErr := engine.TranslateResponse(backend.WireFormat, plan.SourceFormat, body, func(result *llmprotocol.Response) error {
		result.Model = plan.ModelID
		return nil
	})
	if transformResponseErr != nil {
		return nil, transformResponseErr
	}
	if err := i.finalizeResponse(ctx, plan, attempt, responseTerminal(translated.Response)); err != nil {
		return nil, err
	}
	response.Body = io.NopCloser(bytes.NewReader(translated.Body))
	response.ContentLength = int64(len(translated.Body))
	response.Header.Set("Content-Length", strconv.Itoa(len(translated.Body)))
	response.Header.Set("Content-Type", "application/json")
	return response, nil
}

func (i *Invoker) finalizeResponse(
	ctx context.Context,
	plan Plan,
	attempt AttemptResult,
	terminal ResponseTerminal,
) error {
	if i == nil || i.Finalizer == nil {
		return nil
	}
	return i.Finalizer.Finalize(context.WithoutCancel(ctx), plan, attempt, terminal)
}

func responseTerminal(response llmprotocol.Response) ResponseTerminal {
	usage := response.Usage
	if usage.State == "" {
		usage.State = llmprotocol.UsageUnavailable
	}
	return ResponseTerminal{Usage: usage, StopReason: response.StopReason, Error: response.Error}
}

func safeUpstreamHTTPError(status int, headers http.Header) *llmprotocol.ProtocolError {
	protocolError := llmprotocol.NewError(
		llmprotocol.ErrorUpstreamUnavailable, "upstream_unavailable", "the selected model is temporarily unavailable", nil,
	)
	switch status {
	case http.StatusBadRequest, http.StatusUnprocessableEntity:
		protocolError.Category, protocolError.Code, protocolError.Message = llmprotocol.ErrorInvalidRequest, "upstream_invalid_request", "the selected model rejected the request"
	case http.StatusUnauthorized:
		protocolError.Category, protocolError.Code, protocolError.Message = llmprotocol.ErrorAuthentication, "upstream_authentication", "the selected model could not authenticate the request"
	case http.StatusForbidden:
		protocolError.Category, protocolError.Code, protocolError.Message = llmprotocol.ErrorPermission, "upstream_permission", "the selected model denied the request"
	case http.StatusNotFound:
		protocolError.Category, protocolError.Code, protocolError.Message = llmprotocol.ErrorNotFound, "upstream_not_found", "the selected model or endpoint was not found"
	case http.StatusConflict:
		protocolError.Category, protocolError.Code, protocolError.Message = llmprotocol.ErrorConflict, "upstream_conflict", "the selected model reported a request conflict"
	case http.StatusRequestTimeout, http.StatusGatewayTimeout:
		protocolError.Category, protocolError.Code, protocolError.Message = llmprotocol.ErrorUpstreamTimeout, "upstream_timeout", "the selected model timed out"
	case http.StatusTooManyRequests:
		protocolError.Category, protocolError.Code, protocolError.Message = llmprotocol.ErrorRateLimited, "upstream_rate_limited", "the selected model is rate limited"
	}
	if retryAfter, err := strconv.ParseInt(strings.TrimSpace(headers.Get("Retry-After")), 10, 64); err == nil && retryAfter >= 0 && retryAfter <= 86_400 {
		protocolError.RetryAfter = retryAfter
	}
	return protocolError
}

func readWireBody(source io.ReadCloser, limit int) ([]byte, error) {
	defer source.Close()
	body, err := io.ReadAll(io.LimitReader(source, int64(limit)+1))
	if err != nil {
		return nil, err
	}
	if len(body) > limit {
		return nil, fmt.Errorf("backend response exceeds %d bytes", limit)
	}
	return body, nil
}

func applyConnectionHeaders(target, source http.Header) {
	for name, values := range source {
		target.Del(name)
		for _, value := range values {
			target.Add(name, value)
		}
	}
}

type codecStreamBody struct {
	source           io.ReadCloser
	engine           *protocolcodec.StreamEngine
	finalize         func(ResponseTerminal) error
	readBuffer       []byte
	pending          []byte
	semanticTerminal *ResponseTerminal
	terminal         bool
	terminalErr      error
	errorReturned    bool
	finalizeOnce     sync.Once
}

func newCodecStreamBody(
	source io.ReadCloser,
	engine *protocolcodec.StreamEngine,
	finalize func(ResponseTerminal) error,
) io.ReadCloser {
	return &codecStreamBody{
		source: source, engine: engine, finalize: finalize,
		readBuffer: make([]byte, 32<<10),
	}
}

func (body *codecStreamBody) Read(target []byte) (int, error) {
	if len(target) == 0 {
		return 0, nil
	}
	for len(body.pending) == 0 && !body.terminal {
		read, readErr := body.source.Read(body.readBuffer)
		if read > 0 {
			frames, events, _, transformErr := body.engine.Push(body.readBuffer[:read])
			body.appendFrames(frames)
			body.observe(events)
			if transformErr != nil {
				body.finish(transformErr)
			}
		}
		if readErr != nil && !body.terminal {
			reason := readErr
			if readErr == io.EOF {
				reason = nil
			}
			body.finish(reason)
			if readErr != io.EOF {
				body.rememberError(readErr)
			}
		}
	}
	if len(body.pending) == 0 {
		if body.terminalErr != nil && !body.errorReturned {
			body.errorReturned = true
			return 0, body.terminalErr
		}
		return 0, io.EOF
	}
	read := copy(target, body.pending)
	body.pending = body.pending[read:]
	return read, nil
}

func (body *codecStreamBody) Close() error {
	if !body.terminal {
		body.finish(context.Canceled)
	}
	body.terminal = true
	body.pending = nil
	body.readBuffer = nil
	return errors.Join(body.terminalErr, body.source.Close())
}

func (body *codecStreamBody) appendFrames(frames [][]byte) {
	for _, encoded := range frames {
		body.pending = append(body.pending, encoded...)
	}
}

func (body *codecStreamBody) observe(events []llmprotocol.Event) {
	for _, event := range events {
		if event.Type != llmprotocol.EventResponseCompleted && event.Type != llmprotocol.EventResponseFailed {
			continue
		}
		usage := llmprotocol.Usage{State: llmprotocol.UsageUnavailable}
		if event.Usage != nil {
			usage = *event.Usage
		}
		terminal := ResponseTerminal{Usage: usage, StopReason: event.StopReason, Error: event.Error}
		body.semanticTerminal = &terminal
	}
}

func (body *codecStreamBody) finish(reason error) {
	if body.terminal {
		return
	}
	frames, events, _, err := body.engine.Finalize(reason)
	body.appendFrames(frames)
	body.observe(events)
	terminal := body.semanticTerminal
	if err != nil {
		body.rememberError(err)
		terminal = incompleteStreamTerminal(err)
	} else if reason != nil && (terminal == nil || terminal.Error == nil) {
		body.rememberError(reason)
		terminal = incompleteStreamTerminal(reason)
	}
	if terminal == nil {
		terminal = incompleteStreamTerminal(reason)
	}
	body.finalizeOnce.Do(func() {
		if body.finalize != nil {
			body.rememberError(body.finalize(*terminal))
		}
	})
	body.terminal = true
}

func incompleteStreamTerminal(reason error) *ResponseTerminal {
	return &ResponseTerminal{
		Usage:      llmprotocol.Usage{State: llmprotocol.UsageUnavailable},
		StopReason: llmprotocol.StopError,
		Error: llmprotocol.NewError(
			llmprotocol.ErrorUpstreamUnavailable,
			"stream_incomplete",
			"upstream stream ended before completion",
			reason,
		),
	}
}

func (body *codecStreamBody) rememberError(err error) {
	if err != nil {
		body.terminalErr = errors.Join(body.terminalErr, err)
	}
}
