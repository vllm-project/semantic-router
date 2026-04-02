package extproc

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
)

func TestBuildResponseHeaderMutation_IncludesExtendedMatchedSignalHeaders(t *testing.T) {
	ctx := &RequestContext{
		VSRMatchedKeywords:   []string{"keyword:math"},
		VSRMatchedEmbeddings: []string{"embedding:math"},
		VSRMatchedDomains:    []string{"domain:math"},
		VSRMatchedContext:    []string{"context:long"},
		VSRMatchedComplexity: []string{"complexity:hard"},
		VSRMatchedModality:   []string{"AR"},
		VSRMatchedAuthz:      []string{"authz:premium"},
		VSRMatchedJailbreak:  []string{"jailbreak:block"},
		VSRMatchedPII:        []string{"pii:email"},
		VSRMatchedReask:      []string{"likely_dissatisfied"},
		VSRMatchedProjection: []string{"balance_reasoning"},
	}

	mutation := buildResponseHeaderMutation(ctx, true)
	require.NotNil(t, mutation)

	headerMap := map[string]string{}
	for _, header := range mutation.SetHeaders {
		headerMap[header.Header.Key] = string(header.Header.RawValue)
	}

	assert.Equal(t, "complexity:hard", headerMap[headers.VSRMatchedComplexity])
	assert.Equal(t, "AR", headerMap[headers.VSRMatchedModality])
	assert.Equal(t, "authz:premium", headerMap[headers.VSRMatchedAuthz])
	assert.Equal(t, "jailbreak:block", headerMap[headers.VSRMatchedJailbreak])
	assert.Equal(t, "pii:email", headerMap[headers.VSRMatchedPII])
	assert.Equal(t, "likely_dissatisfied", headerMap[headers.VSRMatchedReask])
	assert.Equal(t, "balance_reasoning", headerMap[headers.VSRMatchedProjection])
}
