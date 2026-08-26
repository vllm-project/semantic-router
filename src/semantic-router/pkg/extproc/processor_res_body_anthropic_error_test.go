package extproc

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestNormalizeProviderResponseBody_PreservesAnthropicErrorsPerClientProtocol(t *testing.T) {
	const upstreamError = `{"type":"error","error":{"type":"authentication_error","message":"API key is invalid."}}`

	tests := []struct {
		name            string
		clientProtocol  string
		wantBody        string
		wantTransformed bool
	}{
		{
			name:            "openai client receives openai error shape",
			wantBody:        `{"error":{"type":"authentication_error","message":"API key is invalid."}}`,
			wantTransformed: true,
		},
		{
			name:            "anthropic client receives upstream envelope",
			clientProtocol:  config.ClientProtocolAnthropic,
			wantBody:        upstreamError,
			wantTransformed: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			router := &OpenAIRouter{}
			ctx := &RequestContext{
				APIFormat:      config.APIFormatAnthropic,
				ClientProtocol: tt.clientProtocol,
				RequestModel:   "claude-sonnet-4-5",
			}

			body, transformed, err := router.normalizeProviderResponseBody([]byte(upstreamError), ctx)
			require.NoError(t, err)
			require.Equal(t, tt.wantTransformed, transformed)
			require.JSONEq(t, tt.wantBody, string(body))
		})
	}
}

func TestNormalizeProviderResponseBody_RejectsMalformedJSON(t *testing.T) {
	router := &OpenAIRouter{}
	ctx := &RequestContext{APIFormat: config.APIFormatAnthropic}

	_, _, err := router.normalizeProviderResponseBody([]byte(`{"type":"error"`), ctx)
	require.Error(t, err)

	var syntaxErr *json.SyntaxError
	require.ErrorAs(t, err, &syntaxErr)
}
