package handlers

import (
	"context"
	"crypto/tls"
	"errors"
	"net"
	"net/http"
	"net/netip"
	"net/url"
	"strings"
	"time"
)

const (
	outboundMaxRedirects           = 5
	outboundDefaultTimeout         = 30 * time.Second
	outboundMaxResponseHeaderBytes = 1 * 1024 * 1024
	outboundMaxRequestBodyBytes    = 64 * 1024
	outboundMaxURLBytes            = 4096
)

var (
	errOutboundURLInvalid         = errors.New("outbound URL is invalid")
	errOutboundDestinationBlocked = errors.New("outbound destination is not public")
	errOutboundResolutionFailed   = errors.New("outbound destination resolution failed")
	errOutboundConnectionFailed   = errors.New("outbound connection failed")
	errOutboundRequestFailed      = errors.New("outbound request failed")
	errOutboundRequestTimeout     = errors.New("outbound request timed out")
	errOutboundTooManyRedirects   = errors.New("too many outbound redirects")
	errOutboundResponseTooLarge   = errors.New("outbound response exceeds the size limit")
)

// outboundHTTPClient is deliberately narrower than *http.Client. Production
// handlers receive a public-only implementation; tests that need a loopback
// server must opt in through an explicit controlled implementation.
type outboundHTTPClient interface {
	ValidateURL(context.Context, string) error
	Do(*http.Request) (*http.Response, error)
}

type outboundResolver interface {
	LookupNetIP(context.Context, string, string) ([]netip.Addr, error)
}

type outboundDialContext func(context.Context, string, string) (net.Conn, error)

type outboundHTTPClientOptions struct {
	resolver    outboundResolver
	dialContext outboundDialContext
}

type publicOutboundHTTPClient struct {
	client *http.Client
	policy *outboundNetworkPolicy
}

type outboundNetworkPolicy struct {
	resolver    outboundResolver
	dialContext outboundDialContext
}

type publicOutboundRoundTripper struct {
	transport *http.Transport
	policy    *outboundNetworkPolicy
}

type resolvedOutboundTarget struct {
	host      string
	addresses []netip.Addr
}

type resolvedOutboundTargetContextKey struct{}

func newPublicOutboundHTTPClient(timeout time.Duration) *publicOutboundHTTPClient {
	effectiveTimeout := timeout
	if effectiveTimeout <= 0 {
		effectiveTimeout = outboundDefaultTimeout
	}
	dialer := &net.Dialer{
		Timeout:   effectiveTimeout,
		KeepAlive: 30 * time.Second,
	}
	return newPublicOutboundHTTPClientWithOptions(effectiveTimeout, outboundHTTPClientOptions{
		resolver:    net.DefaultResolver,
		dialContext: dialer.DialContext,
	})
}

func newPublicOutboundHTTPClientWithOptions(
	timeout time.Duration,
	options outboundHTTPClientOptions,
) *publicOutboundHTTPClient {
	effectiveTimeout := timeout
	if effectiveTimeout <= 0 {
		effectiveTimeout = outboundDefaultTimeout
	}
	resolver := options.resolver
	if resolver == nil {
		resolver = net.DefaultResolver
	}
	dialContext := options.dialContext
	if dialContext == nil {
		dialer := &net.Dialer{Timeout: effectiveTimeout, KeepAlive: 30 * time.Second}
		dialContext = dialer.DialContext
	}

	policy := &outboundNetworkPolicy{
		resolver:    resolver,
		dialContext: dialContext,
	}

	// Keep every security-relevant transport option private and explicit. In
	// particular, do not clone http.DefaultTransport: it is a mutable global and
	// another package could have enabled an ambient proxy or insecure TLS there.
	transport := &http.Transport{
		Proxy:                  nil,
		DialContext:            policy.dialValidatedTarget,
		ForceAttemptHTTP2:      true,
		MaxIdleConns:           100,
		MaxIdleConnsPerHost:    10,
		IdleConnTimeout:        90 * time.Second,
		TLSHandshakeTimeout:    10 * time.Second,
		ResponseHeaderTimeout:  effectiveTimeout,
		ExpectContinueTimeout:  1 * time.Second,
		MaxResponseHeaderBytes: outboundMaxResponseHeaderBytes,
		TLSClientConfig: &tls.Config{
			MinVersion: tls.VersionTLS12,
		},
	}

	client := &http.Client{
		Timeout: effectiveTimeout,
		Transport: &publicOutboundRoundTripper{
			transport: transport,
			policy:    policy,
		},
	}
	client.CheckRedirect = func(req *http.Request, via []*http.Request) error {
		// via contains the original request followed by every request already
		// made. Allow five followed redirects and reject the sixth.
		if len(via) > outboundMaxRedirects {
			return errOutboundTooManyRedirects
		}
		_, err := policy.resolvePublicTarget(req.Context(), req.URL)
		return err
	}

	return &publicOutboundHTTPClient{client: client, policy: policy}
}

func (c *publicOutboundHTTPClient) ValidateURL(ctx context.Context, rawURL string) error {
	parsed, err := parseOutboundHTTPURL(rawURL)
	if err != nil {
		return err
	}
	_, err = c.policy.resolvePublicTarget(ctx, parsed)
	return err
}

func (c *publicOutboundHTTPClient) Do(req *http.Request) (*http.Response, error) {
	if req == nil {
		return nil, errOutboundRequestFailed
	}
	resp, err := c.client.Do(req)
	if err == nil {
		return resp, nil
	}
	if resp != nil && resp.Body != nil {
		_ = resp.Body.Close()
	}
	return nil, sanitizeOutboundHTTPError(err)
}

func (t *publicOutboundRoundTripper) RoundTrip(req *http.Request) (*http.Response, error) {
	if req == nil || req.URL == nil {
		return nil, errOutboundURLInvalid
	}
	if req.Host != "" && !strings.EqualFold(req.Host, req.URL.Host) {
		return nil, errOutboundURLInvalid
	}
	target, err := t.policy.resolvePublicTarget(req.Context(), req.URL)
	if err != nil {
		return nil, err
	}
	ctx := context.WithValue(req.Context(), resolvedOutboundTargetContextKey{}, target)
	return t.transport.RoundTrip(req.Clone(ctx))
}

func parseOutboundHTTPURL(rawURL string) (*url.URL, error) {
	trimmed := strings.TrimSpace(rawURL)
	if trimmed == "" || len([]byte(trimmed)) > outboundMaxURLBytes {
		return nil, errOutboundURLInvalid
	}

	parsed, err := url.Parse(trimmed)
	if err != nil || parsed.Opaque != "" {
		return nil, errOutboundURLInvalid
	}
	if !strings.EqualFold(parsed.Scheme, "http") && !strings.EqualFold(parsed.Scheme, "https") {
		return nil, errOutboundURLInvalid
	}
	if parsed.Host == "" || parsed.Hostname() == "" || parsed.User != nil {
		return nil, errOutboundURLInvalid
	}
	// Calling Port also validates bracket and port syntax on supported Go
	// versions; url.Parse rejects malformed ports before this point.
	_ = parsed.Port()
	parsed.Scheme = strings.ToLower(parsed.Scheme)
	return parsed, nil
}

func (p *outboundNetworkPolicy) resolvePublicTarget(
	ctx context.Context,
	targetURL *url.URL,
) (resolvedOutboundTarget, error) {
	if targetURL == nil {
		return resolvedOutboundTarget{}, errOutboundURLInvalid
	}
	parsed, err := parseOutboundHTTPURL(targetURL.String())
	if err != nil {
		return resolvedOutboundTarget{}, err
	}

	host := normalizeOutboundHostname(parsed.Hostname())
	if host == "" || isBlockedOutboundHostname(host) {
		return resolvedOutboundTarget{}, errOutboundDestinationBlocked
	}

	if literal, parseErr := netip.ParseAddr(host); parseErr == nil {
		literal = literal.Unmap()
		if !isPublicOutboundIP(literal) {
			return resolvedOutboundTarget{}, errOutboundDestinationBlocked
		}
		return resolvedOutboundTarget{host: host, addresses: []netip.Addr{literal}}, nil
	}
	if isAmbiguousIPv4Hostname(host) {
		return resolvedOutboundTarget{}, errOutboundDestinationBlocked
	}

	addresses, lookupErr := p.resolver.LookupNetIP(ctx, "ip", host)
	if lookupErr != nil || len(addresses) == 0 {
		return resolvedOutboundTarget{}, errOutboundResolutionFailed
	}

	validated := make([]netip.Addr, 0, len(addresses))
	seen := make(map[netip.Addr]struct{}, len(addresses))
	for _, address := range addresses {
		address = address.Unmap()
		if !isPublicOutboundIP(address) {
			return resolvedOutboundTarget{}, errOutboundDestinationBlocked
		}
		if _, ok := seen[address]; ok {
			continue
		}
		seen[address] = struct{}{}
		validated = append(validated, address)
	}
	if len(validated) == 0 {
		return resolvedOutboundTarget{}, errOutboundResolutionFailed
	}
	return resolvedOutboundTarget{host: host, addresses: validated}, nil
}

func (p *outboundNetworkPolicy) dialValidatedTarget(
	ctx context.Context,
	network string,
	address string,
) (net.Conn, error) {
	host, port, err := net.SplitHostPort(address)
	if err != nil || port == "" {
		return nil, errOutboundConnectionFailed
	}
	host = normalizeOutboundHostname(host)

	target, ok := ctx.Value(resolvedOutboundTargetContextKey{}).(resolvedOutboundTarget)
	if !ok || target.host != host {
		target, err = p.resolvePublicHost(ctx, host)
		if err != nil {
			return nil, err
		}
	}

	for _, ip := range target.addresses {
		if network == "tcp4" && !ip.Is4() {
			continue
		}
		if network == "tcp6" && !ip.Is6() {
			continue
		}
		conn, dialErr := p.dialContext(ctx, network, net.JoinHostPort(ip.String(), port))
		if dialErr == nil {
			return conn, nil
		}
		if errors.Is(dialErr, context.Canceled) {
			return nil, context.Canceled
		}
		if errors.Is(dialErr, context.DeadlineExceeded) {
			return nil, errOutboundRequestTimeout
		}
	}
	return nil, errOutboundConnectionFailed
}

func (p *outboundNetworkPolicy) resolvePublicHost(
	ctx context.Context,
	host string,
) (resolvedOutboundTarget, error) {
	authority := host
	if address, err := netip.ParseAddr(host); err == nil && address.Is6() {
		authority = "[" + host + "]"
	}
	return p.resolvePublicTarget(ctx, &url.URL{Scheme: "http", Host: authority})
}

func sanitizeOutboundHTTPError(err error) error {
	if err == nil {
		return nil
	}
	for _, safeErr := range []error{
		errOutboundURLInvalid,
		errOutboundDestinationBlocked,
		errOutboundResolutionFailed,
		errOutboundConnectionFailed,
		errOutboundRequestTimeout,
		errOutboundTooManyRedirects,
	} {
		if errors.Is(err, safeErr) {
			return safeErr
		}
	}
	if errors.Is(err, context.Canceled) {
		return context.Canceled
	}
	if errors.Is(err, context.DeadlineExceeded) {
		return errOutboundRequestTimeout
	}
	var netErr net.Error
	if errors.As(err, &netErr) && netErr.Timeout() {
		return errOutboundRequestTimeout
	}
	return errOutboundRequestFailed
}
