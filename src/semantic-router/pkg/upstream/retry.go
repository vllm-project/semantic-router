package upstream

import (
	"context"
	"net/http"
	"slices"
	"strconv"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
)

// RetryOn is a set of retry conditions. Its members are Envoy's retry_on
// tokens and mean what they mean in Envoy.
type RetryOn uint16

const (
	// Retry5xx retries a 5xx response or an attempt that got no response.
	Retry5xx RetryOn = 1 << iota
	// RetryGatewayError retries a 502, 503 or 504, or no response.
	RetryGatewayError
	// RetryReset retries an attempt that got no response.
	RetryReset
	// RetryResetBeforeRequest retries a failure before the request was sent.
	RetryResetBeforeRequest
	// RetryConnectFailure retries a failure to connect.
	RetryConnectFailure
	// RetryRefusedStream retries a stream the server refused, at once.
	RetryRefusedStream
	// RetryRetriable4xx retries a 409.
	RetryRetriable4xx
	// RetryRetriableStatusCodes retries the policy's listed statuses.
	RetryRetriableStatusCodes
	// RetryEnvoyRateLimited retries a response marked x-envoy-ratelimited.
	RetryEnvoyRateLimited
)

var retryOnTokens = map[string]RetryOn{
	"5xx":                    Retry5xx,
	"gateway-error":          RetryGatewayError,
	"reset":                  RetryReset,
	"reset-before-request":   RetryResetBeforeRequest,
	"connect-failure":        RetryConnectFailure,
	"refused-stream":         RetryRefusedStream,
	"retriable-4xx":          RetryRetriable4xx,
	"retriable-status-codes": RetryRetriableStatusCodes,
	"envoy-ratelimited":      RetryEnvoyRateLimited,
}

// ParseRetryOn reads a comma-separated retry_on value. Like Envoy, it ignores
// tokens it cannot act on, such as the gRPC status conditions.
func ParseRetryOn(value string) RetryOn {
	var on RetryOn
	for _, token := range strings.Split(value, ",") {
		on |= retryOnTokens[strings.TrimSpace(token)]
	}
	return on
}

// RetryPolicy says when and how a call is retried.
type RetryPolicy struct {
	// NumRetries is the most retries after the first attempt.
	NumRetries int
	On         RetryOn
	// RetriableStatusCodes are retried under RetryRetriableStatusCodes.
	RetriableStatusCodes []int
	// BackOffBase and BackOffMax bound the jittered exponential back-off;
	// zero takes Envoy's 25ms and ten times the base.
	BackOffBase time.Duration
	BackOffMax  time.Duration
	// RetryAfterMax makes a retry wait a response's Retry-After seconds up to
	// this bound; zero ignores the header.
	RetryAfterMax time.Duration
}

// RetryBudget caps concurrent retries at a share of the requests in flight
// instead of Breakers.MaxRetries.
type RetryBudget struct {
	// Percent of the active and pending requests; zero takes Envoy's 20.
	Percent float64
	// MinConcurrency is the cap's floor; zero takes Envoy's 3.
	MinConcurrency int
}

const (
	defaultBackOffBase               = 25 * time.Millisecond
	defaultRetryBudgetPercent        = 20
	defaultRetryBudgetMinConcurrency = 3
	// hostSelectionRetries is the template's host_selection_retry_max_attempts:
	// a retry re-picks this many times to avoid endpoints the call tried.
	hostSelectionRetries = 3
)

type retryDecision uint8

const (
	noRetry retryDecision = iota
	retryWithBackOff
	retryNow
)

// decide applies Envoy's retry rules to one attempt's outcome. sent reports
// whether the request reached the endpoint before the failure.
func (p *RetryPolicy) decide(resp *Response, failure *Error, sent bool) retryDecision {
	if failure == nil {
		return p.decideResponse(resp)
	}
	noResponse := p.On&(Retry5xx|RetryGatewayError|RetryReset) != 0
	switch failure.Kind {
	case KindConnectFailure:
		if noResponse || p.On&(RetryConnectFailure|RetryResetBeforeRequest) != 0 {
			return retryWithBackOff
		}
	case KindRefusedStream:
		if noResponse {
			return retryWithBackOff
		}
		if p.On&RetryRefusedStream != 0 {
			return retryNow
		}
	case KindReset:
		if noResponse || (p.On&RetryResetBeforeRequest != 0 && !sent) {
			return retryWithBackOff
		}
	case KindTimeout:
		// A per-try timeout is a reset in Envoy's terms; the total timeout ends
		// the call.
		if noResponse && (failure.Stage == StagePerTry || failure.Stage == StageFirstByte) {
			return retryWithBackOff
		}
	}
	return noRetry
}

func (p *RetryPolicy) decideResponse(resp *Response) retryDecision {
	if resp.Header.Get("X-Envoy-Ratelimited") != "" {
		if p.On&RetryEnvoyRateLimited != 0 {
			return retryWithBackOff
		}
		return noRetry
	}
	code := resp.StatusCode
	switch {
	case p.On&Retry5xx != 0 && code >= 500 && code < 600,
		p.On&RetryGatewayError != 0 && (code == http.StatusBadGateway ||
			code == http.StatusServiceUnavailable || code == http.StatusGatewayTimeout),
		p.On&RetryRetriable4xx != 0 && code == http.StatusConflict,
		p.On&RetryRetriableStatusCodes != 0 && slices.Contains(p.RetriableStatusCodes, code):
		return retryWithBackOff
	}
	return noRetry
}

// retryAfter is the wait a response asks for with Retry-After, when the
// policy honors it and the value is within RetryAfterMax. Like Envoy's
// rate-limited back-off it adds up to half again as jitter; a value above the
// bound is discarded in favor of the ordinary back-off.
func (p *RetryPolicy) retryAfter(resp *Response, rnd Rand) (time.Duration, bool) {
	if p.RetryAfterMax <= 0 || resp == nil {
		return 0, false
	}
	seconds, err := strconv.Atoi(strings.TrimSpace(resp.Header.Get("Retry-After")))
	if err != nil || seconds < 0 || time.Duration(seconds)*time.Second > p.RetryAfterMax {
		return 0, false
	}
	wait := time.Duration(seconds) * time.Second
	if jitter := wait.Milliseconds() / 2; jitter > 0 {
		wait += time.Duration(randBelow(rnd, jitter)) * time.Millisecond
	}
	return wait, true
}

// randBelow draws uniformly from [0, n) for n > 0.
func randBelow(rnd Rand, n int64) int64 {
	return int64(rnd.Uint64()>>1) % n
}

// backOff is Envoy's jittered exponential back-off: the n-th wait is drawn
// uniformly below min(base * 2^n, max), in whole milliseconds.
type backOff struct {
	next, max time.Duration
	rnd       Rand
}

func newBackOff(p *RetryPolicy, rnd Rand) *backOff {
	base := pickDuration(p.BackOffBase, defaultBackOffBase)
	return &backOff{next: base, max: pickDuration(p.BackOffMax, 10*base), rnd: rnd}
}

func (b *backOff) wait() time.Duration {
	interval := b.next
	if b.next < b.max/2 {
		b.next *= 2
	} else {
		b.next = b.max
	}
	ms := interval.Milliseconds()
	if ms == 0 {
		return 0
	}
	return min(time.Duration(randBelow(b.rnd, ms))*time.Millisecond, b.max)
}

// run makes the call's attempts: the first, then retries while the policy
// allows them. All of it happens before Do returns, so nothing is retried
// once the caller holds a response.
func (c *call) run(ctx context.Context) (*Response, *Error) {
	policy := c.policy.Retry
	var waits *backOff
	for n := 0; ; n++ {
		resp, failure := c.try(ctx)
		c.endRetry()
		if policy == nil || n >= policy.NumRetries {
			return resp, failure
		}
		decision := policy.decide(resp, failure, c.sent)
		if decision == noRetry {
			return resp, failure
		}
		release, overflow := c.cluster.breaker.acquireRetry()
		if overflow != nil {
			return resp, failure
		}
		var wait time.Duration
		if decision == retryWithBackOff {
			if after, ok := policy.retryAfter(resp, c.rnd); ok {
				wait = after
			} else {
				if waits == nil {
					waits = newBackOff(policy, c.rnd)
				}
				wait = waits.wait()
			}
		}
		metrics.RecordUpstreamRetry(c.cluster.spec.Name, outcomeLabel(resp, failure))
		c.discard(resp)
		c.retrySlot = release
		if err := c.sleep(ctx, wait); err != nil {
			c.endRetry()
			return nil, err
		}
	}
}

// endRetry frees the retry slot once the retried attempt has an outcome, as
// Envoy holds a retry from its back-off until the retried request's result.
func (c *call) endRetry() {
	if c.retrySlot != nil {
		c.retrySlot()
		c.retrySlot = nil
	}
}

// discard drops a response the call retries instead of returning.
func (c *call) discard(resp *Response) {
	if resp != nil {
		_ = resp.Body.Close()
	}
}

// sleep waits d on the Set's clock, or until the call ends.
func (c *call) sleep(ctx context.Context, d time.Duration) *Error {
	if d <= 0 {
		return nil
	}
	elapsed := make(chan struct{})
	timer := c.clock.AfterFunc(d, func() { close(elapsed) })
	select {
	case <-elapsed:
		return nil
	case <-ctx.Done():
		timer.Stop()
		failure := contextError(ctx)
		failure.Cluster = c.cluster.spec.Name
		return failure
	}
}

func outcomeLabel(resp *Response, failure *Error) string {
	if failure != nil {
		return string(failure.Kind)
	}
	return statusClass(resp.StatusCode)
}
