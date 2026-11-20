package extproc

import (
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
)

// DeepSeek Harness names itself in its User-Agent, deepseek-harness/<version>,
// and sends no session or agent header, so its calls are known by harness
// alone.
const deepSeekHarnessUserAgentPrefix = "deepseek-harness/"

// isDeepSeekHarness reports whether a request comes from DeepSeek Harness.
func isDeepSeekHarness(ctx *RequestContext) bool {
	return strings.HasPrefix(strings.ToLower(strings.TrimSpace(headerValueCI(ctx, headers.UserAgent))), deepSeekHarnessUserAgentPrefix)
}
