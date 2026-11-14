package extproc

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/ratelimit"
)

type recordingLimiter struct{ reported []ratelimit.TokenUsage }

func (l *recordingLimiter) Name() string { return "recording" }
func (l *recordingLimiter) Check(ratelimit.Context) (*ratelimit.Decision, error) {
	return &ratelimit.Decision{Allowed: true}, nil
}

func (l *recordingLimiter) Report(_ ratelimit.Context, usage ratelimit.TokenUsage) error {
	l.reported = append(l.reported, usage)
	return nil
}

// A stream whose provider sends no usable usage still settles its token
// budget, as the buffered path does; skipping the report left the output
// tokens of streamed traffic out of a tokens_per_unit budget.
func TestStreamingUsageWithoutUsageStillReports(t *testing.T) {
	limiter := &recordingLimiter{}
	router := &OpenAIRouter{RateLimiter: ratelimit.NewRateLimitResolver(limiter)}
	ctx := &RequestContext{RateLimitCtx: &ratelimit.Context{UserID: "u", Model: "general"}}
	router.reportSemanticStreamingUsage(ctx, 0, responseUsageMetrics{invalid: true})
	if len(limiter.reported) != 1 {
		t.Fatalf("reports = %v, want one", limiter.reported)
	}

	known := &recordingLimiter{}
	router = &OpenAIRouter{RateLimiter: ratelimit.NewRateLimitResolver(known)}
	router.reportSemanticStreamingUsage(ctx, 0, responseUsageMetrics{promptTokens: 10, completionTokens: 4})
	if len(known.reported) != 1 || known.reported[0].OutputTokens != 4 {
		t.Fatalf("reports = %v, want the streamed output tokens", known.reported)
	}
}
