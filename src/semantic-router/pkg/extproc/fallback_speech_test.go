package extproc

import (
	"bytes"
	"context"
	"fmt"
	"net/http"
	"testing"
	"time"

	typev3 "github.com/envoyproxy/go-control-plane/envoy/type/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/fallback"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// A speech response has no neutral form, so the primary path forwards the
// audio untouched. A candidate that answers 200 with audio must become the
// client's answer the same way, instead of being discarded as untranslatable.
func TestFallbackServesSpeechAudioFromTheCandidate(t *testing.T) {
	router, cfg := setupFallbackTestRouter(t, fallback.DefaultEnabledPolicy())
	cfg.ModelConfig["speech-primary"] = config.ModelParams{
		PreferredEndpoints: []string{"backend-speech-a"},
		APIFormat:          config.APIFormatSpeech,
	}
	cfg.ModelConfig["speech-fallback"] = config.ModelParams{
		PreferredEndpoints: []string{"backend-speech-b"},
		APIFormat:          config.APIFormatSpeech,
	}
	cfg.VLLMEndpoints = append(cfg.VLLMEndpoints,
		config.VLLMEndpoint{Name: "backend-speech-a", Address: "127.0.0.1", Port: 8011, ProviderProfileName: "prof-openai"},
		config.VLLMEndpoint{Name: "backend-speech-b", Address: "127.0.0.1", Port: 8012, ProviderProfileName: "prof-openai"},
	)

	request := speechTestRequest(t, `{"model":"auto","input":"hello there","voice":"alloy","response_format":"wav"}`)
	ctx := &RequestContext{
		RequestID:               "req_speech_fallback",
		RequestModel:            "speech-primary",
		VSRSelectedModel:        "speech-primary",
		VSRSelectedDecisionName: "voice",
		VSREligibleModelRefs:    []config.ModelRef{{Model: "speech-primary"}, {Model: "speech-fallback"}},
		SourceFormat:            llmprotocol.OpenAISpeechV1,
		TargetFormat:            llmprotocol.OpenAISpeechV1,
		SemanticRequest:         request,
		ProcessingStartTime:     time.Now().Add(-10 * time.Millisecond),
		StartTime:               time.Now().Add(-10 * time.Millisecond),
	}
	ctx.UpstreamStatusCode = http.StatusServiceUnavailable

	audio := []byte("RIFF\x24\x00\x00\x00WAVEfmt ")
	router.fallbackCaller = func(_ context.Context, model string, _ []byte, _ map[string]string) ([]byte, int, error) {
		if model != "speech-fallback" {
			return nil, 500, fmt.Errorf("unexpected fallback model: %s", model)
		}
		return audio, http.StatusOK, nil
	}

	resp := router.handleUpstreamTransportError([]byte(`{"error":{"message":"boom","type":"server_error"}}`), ctx)
	imm := resp.GetImmediateResponse()
	if imm == nil {
		t.Fatalf("expected an immediate response from the fallback, got %#v", resp)
	}
	if got := imm.GetStatus().GetCode(); got != typev3.StatusCode_OK {
		t.Fatalf("status = %v, want 200", got)
	}
	if !bytes.Equal(imm.GetBody(), audio) {
		t.Fatalf("body = %q, want the candidate's audio %q", imm.GetBody(), audio)
	}
	if got := headerValuesByName(imm.GetHeaders().GetSetHeaders())["content-type"]; got != "audio/wav" {
		t.Fatalf("content-type = %q, want audio/wav", got)
	}
	if ctx.FallbackRecord == nil || len(ctx.FallbackRecord.Attempts) != 2 {
		t.Fatalf("attempts = %+v, want the primary and the served candidate", ctx.FallbackRecord)
	}
	if served := ctx.FallbackRecord.Attempts[1]; served.Discarded || served.StatusCode != http.StatusOK {
		t.Fatalf("candidate attempt = %+v, want a served 200", served)
	}
	if ctx.FallbackRecord.FinalStatus != "succeeded" {
		t.Fatalf("final status = %q, want succeeded", ctx.FallbackRecord.FinalStatus)
	}
	if got := headerValuesByName(imm.GetHeaders().GetSetHeaders())[headers.VSRResponsePath]; got != headers.ResponsePathFallback {
		t.Fatalf("%s = %q, want %q", headers.VSRResponsePath, got, headers.ResponsePathFallback)
	}
}
