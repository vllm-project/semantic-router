package testcases

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/google/uuid"

	"github.com/vllm-project/semantic-router/e2e/pkg/fixtures"
)

// This is a serialization fixture, not a PDF-reading capability test.
const inlineFileDataURL = "data:application/pdf;base64,JVBERi0xLjQK"

const (
	inlineFileAlias         = "mock/inline-file-responses"
	inlineFileProviderModel = "inline-file-backend"
)

func runResponsesInlineFiles(ctx context.Context, router, provider *fixtures.ServiceSession) error {
	for _, source := range protocolCodecE2EClients {
		if source.path == "/v1/messages" {
			continue
		}
		for _, stream := range []bool{false, true} {
			body := source.request(inlineFileAlias, "inline file contract", stream)
			var content []map[string]any
			if source.path == "/v1/chat/completions" {
				content = []map[string]any{{"type": "text", "text": "inline file contract"}, {"type": "file", "file": map[string]string{"filename": "probe.pdf", "file_data": inlineFileDataURL}}}
				body["messages"] = []map[string]any{{"role": "user", "content": content}}
			} else {
				content = []map[string]any{{"type": "input_text", "text": "inline file contract"}, {"type": "input_file", "filename": "probe.pdf", "file_data": inlineFileDataURL}}
				body["input"] = []map[string]any{{"role": "user", "content": content}}
			}
			sessionID := "inline-file-" + uuid.NewString()
			_, err := sendProtocolMatrixRequestWithHeaders(ctx, router, source.path, body, stream, map[string]string{"x-vsr-test-session-id": sessionID})
			if err != nil {
				return fmt.Errorf("inline file %s stream=%t: %w", source.name, stream, err)
			}
			raw, err := lastProviderSimulatorRequest(ctx, provider, sessionID)
			if err != nil {
				return err
			}
			// HTTP 200 from a permissive mock does not prove correct serialization.
			if contractErr := assertResponsesInlineFile(raw); contractErr != nil {
				return fmt.Errorf("inline file %s stream=%t: %w", source.name, stream, contractErr)
			}
		}
	}
	return nil
}

func assertResponsesInlineFile(raw []byte) error {
	var observation struct {
		Body struct {
			Model string `json:"model"`
			Input []struct {
				Content []struct {
					Type     string `json:"type"`
					Filename string `json:"filename"`
					FileData string `json:"file_data"`
				} `json:"content"`
			} `json:"input"`
		} `json:"body"`
	}
	if err := json.Unmarshal(raw, &observation); err != nil {
		return err
	}
	if observation.Body.Model != inlineFileProviderModel {
		return fmt.Errorf("provider model = %q, want %q", observation.Body.Model, inlineFileProviderModel)
	}
	files := 0
	for _, item := range observation.Body.Input {
		for _, part := range item.Content {
			if part.Type != "input_file" {
				continue
			}
			files++
			if part.FileData != inlineFileDataURL || part.Filename != "probe.pdf" {
				return fmt.Errorf("inline file data URL or filename changed: %+v", part)
			}
		}
	}
	if files != 1 {
		return fmt.Errorf("expected one inline file, got %d", files)
	}
	return nil
}
