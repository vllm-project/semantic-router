package extproc

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"testing"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func TestImmediateJSONResponsesPreserveSupportedStatusCodes(t *testing.T) {
	router := &OpenAIRouter{}
	constructors := map[string]func(int, []byte) *ext_proc.ProcessingResponse{
		"router": func(code int, body []byte) *ext_proc.ProcessingResponse {
			return router.createJSONResponseWithBody(code, body, "")
		},
		"responses_api": createImmediateJSONResponse,
	}
	for name, create := range constructors {
		for _, test := range []struct{ input, want int }{
			{200, 200},
			{201, 201},
			{204, 204},
			{302, 302},
			{400, 400},
			{401, 401},
			{408, 408},
			{409, 409},
			{412, 412},
			{415, 415},
			{429, 429},
			{500, 500},
			{501, 501},
			{504, 504},
			{499, 500},
			{599, 500},
			{-1, 500},
			{0, 500},
			{100, 500},
			{199, 500},
			{600, 500},
			{700, 500},
			{int(^uint(0) >> 1), 500},
		} {
			t.Run(fmt.Sprintf("%s/%d", name, test.input), func(t *testing.T) {
				response := create(test.input, nil).GetImmediateResponse()
				require.EqualValues(t, test.want, response.GetStatus().GetCode())
				require.NoError(t, response.ValidateAll())
			})
		}
	}
}

func TestImmediateResponseErrorStatusNeverBecomesSuccess(t *testing.T) {
	for code := 400; code <= 599; code++ {
		got := statusCodeToImmediateResponseCode(code)
		require.GreaterOrEqual(t, int(got), 400, "input status %d", code)
		require.Less(t, int(got), 600, "input status %d", code)
	}
}

func TestErrorResponsesKeepBodyAndHTTPStatusConsistent(t *testing.T) {
	router := &OpenAIRouter{}
	constructors := map[string]func(int, string) *ext_proc.ProcessingResponse{
		"router":        router.createErrorResponse,
		"responses_api": createResponseAPIError,
	}
	for name, create := range constructors {
		for _, test := range []struct {
			input, want int
			kind        string
		}{
			{400, 400, "invalid_request_error"},
			{415, 415, "invalid_request_error"},
			{499, 500, "api_error"},
			{500, 500, "api_error"},
			{504, 504, "api_error"},
			{700, 500, "api_error"},
			{0, 500, "api_error"},
			{200, 500, "api_error"},
			{302, 500, "api_error"},
		} {
			t.Run(fmt.Sprintf("%s/%d", name, test.input), func(t *testing.T) {
				response := create(test.input, "request failed").GetImmediateResponse()
				var body struct {
					Error struct {
						Code int    `json:"code"`
						Type string `json:"type"`
					} `json:"error"`
				}
				require.NoError(t, json.Unmarshal(response.Body, &body))
				require.EqualValues(t, test.want, response.GetStatus().GetCode())
				require.Equal(t, test.want, body.Error.Code)
				require.Equal(t, test.kind, body.Error.Type)
				require.NoError(t, response.ValidateAll())
			})
		}
	}
}

func TestEncodeImmediateResponsePreservesCancellationAndTimeout(t *testing.T) {
	router := &OpenAIRouter{}
	formats := []llmprotocol.WireFormat{
		llmprotocol.OpenAIChatV1,
		llmprotocol.OpenAIResponsesV1,
		llmprotocol.AnthropicMessagesV1,
	}

	for _, format := range formats {
		t.Run(string(format)+"_timeout", func(t *testing.T) {
			ctx := &RequestContext{
				SourceFormat: format,
				ImmediateProtocolError: llmprotocol.NewError(
					llmprotocol.ErrorUpstreamTimeout,
					"request_timeout",
					"request timed out",
					nil,
				),
			}
			resp := router.createErrorResponse(http.StatusGatewayTimeout, "request timed out")
			encoded := router.encodeImmediateResponseForClient(resp, ctx).GetImmediateResponse()
			if got := int(encoded.GetStatus().GetCode()); got != http.StatusGatewayTimeout {
				t.Fatalf("expected HTTP 504, got %d", got)
			}
			bodyStr := string(encoded.Body)
			if !strings.Contains(bodyStr, "request_timeout") && !strings.Contains(bodyStr, "request timed out") {
				t.Fatalf("expected timeout details in body, got: %s", bodyStr)
			}
		})

		t.Run(string(format)+"_canceled", func(t *testing.T) {
			ctx := &RequestContext{
				SourceFormat: format,
				ImmediateProtocolError: llmprotocol.NewError(
					llmprotocol.ErrorInternal,
					"request_canceled",
					"request canceled",
					nil,
				),
			}
			resp := router.createErrorResponse(http.StatusInternalServerError, "request canceled")
			encoded := router.encodeImmediateResponseForClient(resp, ctx).GetImmediateResponse()
			if got := int(encoded.GetStatus().GetCode()); got != http.StatusInternalServerError {
				t.Fatalf("expected HTTP 500, got %d", got)
			}
			bodyStr := string(encoded.Body)
			if !strings.Contains(bodyStr, "request_canceled") && !strings.Contains(bodyStr, "request canceled") {
				t.Fatalf("expected canceled details in body, got: %s", bodyStr)
			}
		})
	}
}
