package extproc

import (
	"crypto/sha256"
	"fmt"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func TestPreparedProviderRequestOwnsPayloadAndMetadata(t *testing.T) {
	body := []byte(`{"model":"provider-model","messages":[{"role":"user","content":"hello"}]}`)
	request := newPreparedProviderRequest(body, preparedProviderRequestMetadata{
		wireFormat: llmprotocol.OpenAIChatV1,
	})

	require.Equal(t, body, request.body)
	require.Same(t, &body[0], &request.body[0], "construction must transfer payload ownership without another large copy")
	require.Equal(t, llmprotocol.OpenAIChatV1, request.metadata.wireFormat)
	require.Equal(t, len(body), request.byteLength())

	digest := sha256.Sum256(body)
	require.Equal(t, fmt.Sprintf("%x", digest), request.payloadSHA256())
}
