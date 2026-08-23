package backendegress

import (
	"context"
	"crypto/tls"
	"errors"
	"fmt"
	"net"
	"net/http"
	"strconv"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/backendinvoker"
)

type TransportOptions struct {
	Guard            Guard
	AdditionalGuards []Guard
	DialTimeout      time.Duration
	TLSConfig        *tls.Config
}

type Transport struct {
	guard      Guard
	additional []Guard
	base       *http.Transport
	dial       func(context.Context, string, string) (net.Conn, error)
}

type resolvedTargetContextKey struct{}

func NewTransport(options TransportOptions) (*Transport, error) {
	if len(options.Guard.Policy.rules) == 0 {
		return nil, fmt.Errorf("backend egress policy is required")
	}
	for _, guard := range options.AdditionalGuards {
		if len(guard.Policy.rules) == 0 {
			return nil, fmt.Errorf("additional backend egress policy is required")
		}
	}
	timeout := options.DialTimeout
	if timeout == 0 {
		timeout = 10 * time.Second
	}
	if timeout <= 0 {
		return nil, fmt.Errorf("backend dial timeout must be positive")
	}
	tlsConfig := &tls.Config{MinVersion: tls.VersionTLS12}
	if options.TLSConfig != nil {
		tlsConfig = options.TLSConfig.Clone()
		if tlsConfig.MinVersion == 0 || tlsConfig.MinVersion < tls.VersionTLS12 {
			tlsConfig.MinVersion = tls.VersionTLS12
		}
	}
	dialer := &net.Dialer{Timeout: timeout, KeepAlive: 30 * time.Second}
	transport := &Transport{
		guard: options.Guard, additional: append([]Guard(nil), options.AdditionalGuards...),
		dial: dialer.DialContext,
	}
	transport.base = &http.Transport{
		Proxy:                 nil,
		DialContext:           transport.dialContext(),
		ForceAttemptHTTP2:     true,
		TLSClientConfig:       tlsConfig,
		TLSHandshakeTimeout:   10 * time.Second,
		ResponseHeaderTimeout: 0,
		ExpectContinueTimeout: time.Second,
		IdleConnTimeout:       90 * time.Second,
		MaxIdleConns:          100,
		MaxIdleConnsPerHost:   10,
	}
	return transport, nil
}

func (t *Transport) RoundTrip(request *http.Request) (*http.Response, error) {
	if t == nil || t.base == nil || request == nil || request.URL == nil {
		return nil, fmt.Errorf("backend egress transport is not initialized")
	}
	origin := request.URL.Scheme + "://" + request.URL.Host
	resolved, err := t.guard.Resolve(request.Context(), origin)
	if err != nil {
		return nil, knownZeroBeforeRequest(request.Context(), err)
	}
	for _, guard := range t.additional {
		additional, guardErr := guard.Resolve(request.Context(), origin)
		if guardErr != nil {
			return nil, knownZeroBeforeRequest(request.Context(), guardErr)
		}
		if !sameResolvedTarget(resolved, additional) {
			return nil, knownZeroBeforeRequest(request.Context(), fmt.Errorf("egress policy resolutions disagree"))
		}
	}
	clone := request.Clone(context.WithValue(request.Context(), resolvedTargetContextKey{}, resolved))
	return t.base.RoundTrip(clone)
}

func sameResolvedTarget(left, right ResolvedTarget) bool {
	if left.Origin != right.Origin || left.Host != right.Host || left.Port != right.Port ||
		len(left.Addresses) != len(right.Addresses) {
		return false
	}
	for index := range left.Addresses {
		if left.Addresses[index] != right.Addresses[index] {
			return false
		}
	}
	return true
}

func (t *Transport) CloseIdleConnections() { t.base.CloseIdleConnections() }

func (t *Transport) dialContext() func(context.Context, string, string) (net.Conn, error) {
	return func(ctx context.Context, network, authority string) (net.Conn, error) {
		resolved, ok := ctx.Value(resolvedTargetContextKey{}).(ResolvedTarget)
		if !ok {
			return nil, knownZeroBeforeRequest(ctx, fmt.Errorf("backend egress resolution is missing"))
		}
		host, portText, err := net.SplitHostPort(authority)
		if err != nil || host != resolved.Host || portText != strconv.Itoa(int(resolved.Port)) {
			return nil, knownZeroBeforeRequest(ctx, fmt.Errorf("backend dial authority differs from authorized target"))
		}
		var lastErr error
		for _, address := range resolved.Addresses {
			connection, dialErr := t.dial(ctx, network, net.JoinHostPort(address.String(), portText))
			if dialErr == nil {
				return connection, nil
			}
			lastErr = dialErr
		}
		return nil, knownZeroBeforeRequest(ctx, fmt.Errorf("dial authorized backend addresses: %w", lastErr))
	}
}

func knownZeroBeforeRequest(ctx context.Context, cause error) error {
	if cause == nil || errors.Is(cause, context.Canceled) ||
		(ctx != nil && errors.Is(ctx.Err(), context.Canceled)) {
		return cause
	}
	trigger := backendinvoker.FallbackUnavailable
	var networkError net.Error
	if errors.Is(cause, context.DeadlineExceeded) ||
		(errors.As(cause, &networkError) && networkError.Timeout()) ||
		(ctx != nil && errors.Is(ctx.Err(), context.DeadlineExceeded)) {
		trigger = backendinvoker.FallbackTimeout
	}
	return backendinvoker.NewKnownZeroTransportFailure(trigger, cause)
}

func NewHTTPClient(transport http.RoundTripper, secretBearing bool) *http.Client {
	client := &http.Client{Transport: transport}
	if secretBearing {
		client.CheckRedirect = func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }
	}
	return client
}
