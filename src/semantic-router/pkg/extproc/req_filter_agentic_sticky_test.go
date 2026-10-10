package extproc

import (
	"fmt"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/agenticfacts"
)

func TestNonPortableContextBindingFromAgenticFacts(t *testing.T) {
	sticky := func(t *testing.T) *agenticfacts.Accepted {
		t.Helper()
		return acceptedAgenticFactsForTest(t, fmt.Sprintf(
			`{"version":"1","expires_at":%q,"context_portability":"sticky"}`,
			time.Now().Add(2*time.Minute).UTC().Format(time.RFC3339),
		))
	}
	portable := func(t *testing.T) *agenticfacts.Accepted {
		t.Helper()
		return acceptedAgenticFactsForTest(t, fmt.Sprintf(
			`{"version":"1","expires_at":%q,"context_portability":"portable"}`,
			time.Now().Add(2*time.Minute).UTC().Format(time.RFC3339),
		))
	}

	t.Run("nil context", func(t *testing.T) {
		bound, reason := nonPortableContextBinding(nil)
		assert.False(t, bound)
		assert.Empty(t, reason)
	})

	t.Run("no facts and no previous response", func(t *testing.T) {
		bound, reason := nonPortableContextBinding(&RequestContext{})
		assert.False(t, bound)
		assert.Empty(t, reason)
	})

	t.Run("sticky binds with its own reason", func(t *testing.T) {
		bound, reason := nonPortableContextBinding(&RequestContext{
			AgenticFacts: agenticfacts.Result{Accepted: sticky(t)},
		})
		assert.True(t, bound)
		assert.Equal(t, "agentic_facts_sticky", reason)
	})

	t.Run("portable does not bind", func(t *testing.T) {
		bound, reason := nonPortableContextBinding(&RequestContext{
			AgenticFacts: agenticfacts.Result{Accepted: portable(t)},
		})
		assert.False(t, bound)
		assert.Empty(t, reason)
	})

	t.Run("previous_response_id wins over sticky", func(t *testing.T) {
		bound, reason := nonPortableContextBinding(&RequestContext{
			PreviousResponseID: "resp-123",
			AgenticFacts:       agenticfacts.Result{Accepted: sticky(t)},
		})
		assert.True(t, bound)
		assert.Equal(t, "previous_response_id", reason,
			"provider-side state is a protocol fact and names the more specific cause")
	})
}
