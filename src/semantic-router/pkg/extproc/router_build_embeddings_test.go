package extproc

import (
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestBuildRouterOwnsPreparedEmbeddings(t *testing.T) {
	components, err := buildRouterComponents(&config.RouterConfig{})
	require.NoError(t, err)
	router := components.buildRouter()
	defer func() { require.NoError(t, router.Close()) }()

	require.NotNil(t, components.embeddings)
	require.Same(t, components.embeddings, router.Embeddings)
}
