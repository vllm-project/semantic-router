package upstream

import (
	"slices"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routing"
)

// NoTimeout disables a timeout explicitly. A zero duration inherits the
// value of the layer below instead.
const NoTimeout time.Duration = -1

const (
	// defaultConnectTimeout is the connect_timeout of every model cluster in
	// the Envoy template.
	defaultConnectTimeout = 10 * time.Second
	// defaultRouteTimeout is the listener timeout the CLI applies when a
	// listener sets none; Envoy uses it as both the route timeout and the
	// stream idle timeout.
	defaultRouteTimeout = 300 * time.Second
)

// Timeouts bound one upstream call. Each layer of policy (built-in defaults,
// listener, cluster, then the caller's override) replaces only the fields it
// sets.
type Timeouts struct {
	// Connect bounds opening a connection, the TLS handshake included. It
	// belongs to the cluster's connection pool, so only the cluster layer
	// sets it; listener and call layers cannot.
	Connect time.Duration
	// FirstByte bounds the wait, after the request is sent, for the first
	// response body byte. Do holds the response until that byte arrives, so a
	// stalled stream fails before anything reaches the client.
	FirstByte time.Duration
	// PerTry bounds one attempt until its response is ready to commit.
	PerTry time.Duration
	// Total bounds the whole call: every attempt and the streamed body.
	Total time.Duration
	// Idle bounds each wait for more of the response body.
	Idle time.Duration
}

// Policy is one layer of per-call behavior. Zero fields inherit.
type Policy struct {
	Timeouts Timeouts
	// Retry, when set, replaces the retry policy of the layers below as a
	// whole, as a route's retry policy replaces its virtual host's in Envoy.
	Retry *RetryPolicy
}

func builtinPolicy() Policy {
	return Policy{Timeouts: Timeouts{
		Connect: defaultConnectTimeout,
		Total:   defaultRouteTimeout,
		Idle:    defaultRouteTimeout,
	}}
}

// over returns p with every unset field taken from base.
func (p Policy) over(base Policy) Policy {
	retry := p.Retry
	if retry == nil {
		retry = base.Retry
	}
	return Policy{Timeouts: p.Timeouts.over(base.Timeouts), Retry: retry}
}

func (t Timeouts) over(base Timeouts) Timeouts {
	return Timeouts{
		Connect:   pickDuration(t.Connect, base.Connect),
		FirstByte: pickDuration(t.FirstByte, base.FirstByte),
		PerTry:    pickDuration(t.PerTry, base.PerTry),
		Total:     pickDuration(t.Total, base.Total),
		Idle:      pickDuration(t.Idle, base.Idle),
	}
}

func pickDuration(value, base time.Duration) time.Duration {
	if value != 0 {
		return value
	}
	return base
}

// merge applies a call's reliability overrides in turn, the lowest first: a
// request-graph step's override is one more layer over its decision's.
func (p Policy) merge(layers ...*routing.Reliability) Policy {
	for _, layer := range layers {
		if layer != nil {
			p = p.mergeLayer(layer)
		}
	}
	return p
}

// mergeLayer applies one override the way Envoy's router applies the
// per-request headers that the Envoy-mode router sends for it: timeouts and
// the retry count replace, retry conditions and retriable status codes add.
// Retries set over a policy without any start, as in Envoy, from one retry,
// and on the default conditions when the override names none.
func (p Policy) mergeLayer(r *routing.Reliability) Policy {
	for _, field := range []struct{ out, in *time.Duration }{
		{&p.Timeouts.Total, r.TotalTimeout},
		{&p.Timeouts.PerTry, r.PerTryTimeout},
		{&p.Timeouts.Idle, r.IdleTimeout},
		{&p.Timeouts.FirstByte, r.FirstByteTimeout},
	} {
		if field.in != nil {
			*field.out = *field.in
			if *field.out == 0 {
				*field.out = NoTimeout
			}
		}
	}
	if r.RetryCount == nil && len(r.RetryOn) == 0 && len(r.RetriableStatusCodes) == 0 &&
		r.RetryBackOffBase == nil && r.RetryBackOffMax == nil && r.RetryAfterMax == nil {
		return p
	}
	retry := RetryPolicy{NumRetries: 1}
	if p.Retry != nil {
		retry = *p.Retry
	}
	if r.RetryCount != nil {
		retry.NumRetries = *r.RetryCount
	}
	retry.On |= ParseRetryOn(strings.Join(r.RetryOn, ","))
	if retry.On == 0 {
		retry.On = ParseRetryOn(config.DefaultProviderRetryOn)
	}
	retry.RetriableStatusCodes = append(slices.Clone(retry.RetriableStatusCodes), r.RetriableStatusCodes...)
	for _, field := range []struct{ out, in *time.Duration }{
		{&retry.BackOffBase, r.RetryBackOffBase},
		{&retry.BackOffMax, r.RetryBackOffMax},
		{&retry.RetryAfterMax, r.RetryAfterMax},
	} {
		if field.in != nil {
			*field.out = *field.in
		}
	}
	p.Retry = &retry
	return p
}

// enabled reports whether d is a positive, active timeout.
func enabled(d time.Duration) bool {
	return d > 0
}
