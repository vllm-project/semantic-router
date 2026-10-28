package extproc

import (
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func TestCloneSemanticRequestForReplayPreservesAbsentRawFields(t *testing.T) {
	request := testNeutralRequest("model-primary", "hello")
	request.Tools = []llmprotocol.Tool{{Name: "lookup"}}

	cloned, err := cloneSemanticRequestForReplay(request)
	require.NoError(t, err)
	require.Nil(t, cloned.OutputFormat.Schema)
	require.Nil(t, cloned.ChatTemplateKwargs)
	require.Nil(t, cloned.ContextManagement)
	require.Nil(t, cloned.Tools[0].InputSchema)
}
