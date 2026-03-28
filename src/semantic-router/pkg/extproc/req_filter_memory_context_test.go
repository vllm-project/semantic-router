package extproc

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestInjectMemoryMessages_InsertsAfterSystemAndDeveloperMessages(t *testing.T) {
	requestBody := []byte(`{
		"messages": [
			{"role": "system", "content": "system instructions"},
			{"role": "developer", "content": "developer instructions"},
			{"role": "user", "content": "hello"}
		]
	}`)

	modified, err := injectMemoryMessages(requestBody, "memory context")
	require.NoError(t, err)

	var request struct {
		Messages []struct {
			Role    string `json:"role"`
			Content string `json:"content"`
		} `json:"messages"`
	}
	require.NoError(t, json.Unmarshal(modified, &request))
	require.Len(t, request.Messages, 4)
	assert.Equal(t, "system", request.Messages[0].Role)
	assert.Equal(t, "developer", request.Messages[1].Role)
	assert.Equal(t, "user", request.Messages[2].Role)
	assert.Equal(t, "memory context", request.Messages[2].Content)
	assert.Equal(t, "user", request.Messages[3].Role)
	assert.Equal(t, "hello", request.Messages[3].Content)
}

func TestInjectMemoryMessages_InitializesMissingMessages(t *testing.T) {
	modified, err := injectMemoryMessages([]byte(`{"model":"test-model"}`), "memory context")
	require.NoError(t, err)

	var request struct {
		Messages []struct {
			Role    string `json:"role"`
			Content string `json:"content"`
		} `json:"messages"`
	}
	require.NoError(t, json.Unmarshal(modified, &request))
	require.Len(t, request.Messages, 1)
	assert.Equal(t, "user", request.Messages[0].Role)
	assert.Equal(t, "memory context", request.Messages[0].Content)
}
