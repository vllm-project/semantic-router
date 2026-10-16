package testcases

import (
	"context"
	"fmt"
	"net/http"
	"strings"

	"k8s.io/client-go/kubernetes"

	"github.com/vllm-project/semantic-router/e2e/pkg/fixtures"
	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

const (
	openRouterReplyMarker = "__mock_openrouter_reply__"
	openRouterReplyAnswer = "OpenRouter fixture answer"
)

func init() {
	pkgtestcases.Register("protocol-codec-openrouter-reply", pkgtestcases.TestCase{
		Description: "OpenRouter Chat response decorations and repeated finish chunks translate to every client format",
		Tags:        []string{"protocol-codec", "response-api", "provider", "streaming"},
		Fn:          testProtocolCodecOpenRouterReply,
	})
}

func testProtocolCodecOpenRouterReply(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	session, err := fixtures.OpenServiceSession(ctx, client, opts)
	if err != nil {
		return err
	}
	defer session.Close()
	provider, err := openProtocolCodecProviderSession(ctx, client, opts, "openai.chat.v1")
	if err != nil {
		return err
	}
	defer provider.Close()

	results := make(map[string]string, 2*len(protocolCodecE2EClients))
	for _, clientContract := range protocolCodecE2EClients {
		for _, streamed := range []bool{false, true} {
			mode := "buffered"
			if streamed {
				mode = "streamed"
			}
			cell := clientContract.name + "/" + mode
			sessionID := "openrouter-reply-" + clientContract.name + "-" + mode
			result, requestErr := sendProtocolMatrixRaw(ctx, session, clientContract.path,
				clientContract.request(chatBackendModel, openRouterReplyMarker, streamed), streamed,
				map[string]string{"x-vsr-test-session-id": sessionID})
			if requestErr != nil {
				return fmt.Errorf("%s request: %w", cell, requestErr)
			}
			if result.StatusCode != http.StatusOK {
				return fmt.Errorf("%s returned HTTP %d: %s", cell, result.StatusCode, truncateString(string(result.Body), 500))
			}
			if streamed {
				if err := clientContract.validateStream(result.Body, openRouterReplyAnswer); err != nil {
					return fmt.Errorf("%s stream: %w", cell, err)
				}
				text, err := extractProtocolStructuredOutputStreamText(clientContract.path, result.Body)
				if err != nil {
					return fmt.Errorf("%s stream text: %w", cell, err)
				}
				if text != openRouterReplyAnswer {
					return fmt.Errorf("%s duplicated or changed output after repeated terminal chunks: %q", cell, text)
				}
				if clientContract.path == "/v1/chat/completions" &&
					(strings.Count(string(result.Body), `"finish_reason":"stop"`) != 1 ||
						strings.Count(string(result.Body), "data: [DONE]") != 1) {
					return fmt.Errorf("%s emitted repeated Chat terminal chunks: %s", cell, truncateString(string(result.Body), 600))
				}
			} else {
				if err := clientContract.validateBuffered(result.Body, openRouterReplyAnswer); err != nil {
					return fmt.Errorf("%s response: %w", cell, err)
				}
				warnings := result.Headers.Get("x-vsr-protocol-warnings")
				for _, field := range []string{
					"provider", "choices.native_finish_reason", "usage.cost", "usage.is_byok",
					"usage.cost_details", "usage.server_tool_use", "usage.prompt_tokens_details.video_tokens",
					"usage.completion_tokens_details.image_tokens",
				} {
					if !hasProtocolFieldDiagnostic(warnings, "dropped", field) {
						return fmt.Errorf("%s omitted %s without a diagnostic: %q", cell, field, warnings)
					}
				}
			}
			forbiddenFields := []string{
				`"provider":`, `"native_finish_reason":`, `"cost":`, `"cost_details":`, `"is_byok":`,
				`"video_tokens":`, `"image_tokens":`,
			}
			if clientContract.path != "/v1/messages" {
				forbiddenFields = append(forbiddenFields, `"server_tool_use":`)
			}
			for _, field := range forbiddenFields {
				if strings.Contains(string(result.Body), field) {
					return fmt.Errorf("%s leaked OpenRouter-only field %s: %s", cell, field, truncateString(string(result.Body), 600))
				}
			}
			if err := verifyProviderSimulatorRequest(ctx, provider, sessionID, "openai.chat.v1", openRouterReplyMarker); err != nil {
				return fmt.Errorf("%s provider dispatch: %w", cell, err)
			}
			results[cell] = "passed"
		}
	}
	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{"matrix_cells": results, "backend_format": "openai.chat.v1"})
	}
	return nil
}
