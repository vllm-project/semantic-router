package safefetch

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/netip"
	"net/url"
	"strings"
	"time"
)

// Sentinel errors. Callers map these onto their own response shapes with
// errors.Is rather than matching on message text.
var (
	// ErrInvalidURL is a URL that is not absolute, has no host, or carries
	// credentials or a fragment.
	ErrInvalidURL = errors.New("safefetch: url is invalid")
	// ErrSchemeNotAllowed is a scheme outside the policy's allowed set.
	ErrSchemeNotAllowed = errors.New("safefetch: url scheme is not allowed")
	// ErrDestinationForbidden is a host that does not resolve, or resolves to
	// an address the policy refuses to dial.
	ErrDestinationForbidden = errors.New("safefetch: destination is not a public address")
	// ErrTooManyRedirects is a redirect chain longer than the policy allows.
	ErrTooManyRedirects = errors.New("safefetch: too many redirects")
	// ErrResponseTooLarge is a body that exceeds the decoded-response budget.
	ErrResponseTooLarge = errors.New("safefetch: response exceeds the size limit")
)

// IPResolver resolves a host to addresses. Tests substitute one to drive
// address classes and mid-flight answer changes without real DNS.
type IPResolver interface {
	LookupNetIP(ctx context.Context, network, host string) ([]netip.Addr, error)
}

// Policy is the outbound-fetch contract for one caller-supplied URL.
//
// The zero value is not usable; start from DefaultPolicy and adjust.
type Policy struct {
	// AllowedSchemes are the URL schemes this consumer accepts, lowercase.
	AllowedSchemes []string
	// MaxRedirects is the number of redirects followed before refusing.
	MaxRedirects int
	// Timeout bounds the whole request, including redirects and body read.
	Timeout time.Duration
	// DialTimeout bounds one connection attempt.
	DialTimeout time.Duration
	// TLSHandshakeTimeout bounds the TLS handshake.
	TLSHandshakeTimeout time.Duration
	// ResponseHeaderTimeout bounds the wait for response headers, so a peer
	// that accepts the connection and then stalls cannot hold it open.
	ResponseHeaderTimeout time.Duration
	// Resolver looks up destination addresses. Nil means net.DefaultResolver.
	Resolver IPResolver
	// AllowedPrivatePrefixes are the only destinations exempt from the
	// public-address requirement. Empty by default: a private target is a
	// deliberate operator decision, declared as a narrow prefix, never a
	// blanket "skip the check" switch.
	AllowedPrivatePrefixes []netip.Prefix
}

// DefaultPolicy is the public-web baseline: HTTPS and HTTP, bounded redirects,
// and deadlines on every stage. Callers tighten it; none may loosen the
// address check, which is not configurable.
func DefaultPolicy() Policy {
	return Policy{
		AllowedSchemes:        []string{"https", "http"},
		MaxRedirects:          5,
		Timeout:               30 * time.Second,
		DialTimeout:           10 * time.Second,
		TLSHandshakeTimeout:   10 * time.Second,
		ResponseHeaderTimeout: 15 * time.Second,
	}
}

// WithSchemes returns a copy of p accepting only the given schemes.
func (p Policy) WithSchemes(schemes ...string) Policy {
	p.AllowedSchemes = schemes
	return p
}

// WithTimeout returns a copy of p with a different overall deadline.
func (p Policy) WithTimeout(timeout time.Duration) Policy {
	p.Timeout = timeout
	return p
}

// WithResolver returns a copy of p using the given resolver.
func (p Policy) WithResolver(resolver IPResolver) Policy {
	p.Resolver = resolver
	return p
}

// AllowingPrivate returns a copy of p that additionally permits destinations
// inside the given prefixes. Use it for a declared internal target, not to
// widen the default policy.
func (p Policy) AllowingPrivate(prefixes ...netip.Prefix) Policy {
	p.AllowedPrivatePrefixes = prefixes
	return p
}

// destinationAllowed reports whether one resolved address may be dialled:
// public, or inside an explicitly declared private prefix.
func (p Policy) destinationAllowed(address netip.Addr) bool {
	if IsPublicAddr(address) {
		return true
	}
	unmapped := address.Unmap()
	for _, prefix := range p.AllowedPrivatePrefixes {
		if prefix.Contains(unmapped) {
			return true
		}
	}
	return false
}

// ValidateURL parses raw and checks everything decidable before DNS: the URL
// is absolute and hierarchical, the scheme is allowed, and it carries no
// credentials or fragment.
//
// Passing this is necessary but not sufficient. The destination address is
// checked again at dial time, because only then is it known.
func (p Policy) ValidateURL(raw string) (*url.URL, error) {
	parsed, err := url.Parse(strings.TrimSpace(raw))
	if err != nil {
		return nil, fmt.Errorf("%w: %w", ErrInvalidURL, err)
	}
	if !parsed.IsAbs() || parsed.Opaque != "" {
		return nil, ErrInvalidURL
	}
	// Scheme first, so a refused scheme reports as one. file:// and friends
	// also have no host, and reporting that instead hides the real reason.
	if !p.schemeAllowed(parsed.Scheme) {
		return nil, fmt.Errorf("%w: %q", ErrSchemeNotAllowed, parsed.Scheme)
	}
	if parsed.Host == "" || parsed.Hostname() == "" {
		return nil, ErrInvalidURL
	}
	// Credentials would be replayed to a redirect target; a fragment is never
	// sent and its presence means the caller built the URL by concatenation.
	if parsed.User != nil || parsed.Fragment != "" {
		return nil, ErrInvalidURL
	}
	return parsed, nil
}

func (p Policy) schemeAllowed(scheme string) bool {
	scheme = strings.ToLower(scheme)
	for _, allowed := range p.AllowedSchemes {
		if scheme == allowed {
			return true
		}
	}
	return false
}

// NewClient builds an http.Client that enforces p.
//
// The transport resolves the host itself, refuses the request unless every
// returned address is public, and then dials one of those resolved addresses
// literally. Dialing the address rather than the name closes the check-to-dial
// window: a resolver that answers publicly during validation and privately
// afterwards cannot move the connection, because the address was already
// chosen. Redirects are revalidated against the same policy.
func (p Policy) NewClient() *http.Client {
	resolver := p.Resolver
	if resolver == nil {
		resolver = net.DefaultResolver
	}
	dialer := &net.Dialer{Timeout: p.DialTimeout, KeepAlive: 30 * time.Second}

	transport := &http.Transport{
		// No proxy: a proxy would terminate the connection somewhere this
		// policy never inspected, making the address check meaningless.
		Proxy: nil,
		// Bound the response by decoded bytes, so compression cannot be used
		// to smuggle a body past the caller's limit.
		DisableCompression:    true,
		ForceAttemptHTTP2:     true,
		TLSHandshakeTimeout:   p.TLSHandshakeTimeout,
		ResponseHeaderTimeout: p.ResponseHeaderTimeout,
		DialContext: func(ctx context.Context, network, address string) (net.Conn, error) {
			return p.dial(ctx, resolver, dialer, network, address)
		},
	}

	return &http.Client{
		Transport: transport,
		Timeout:   p.Timeout,
		CheckRedirect: func(request *http.Request, via []*http.Request) error {
			if len(request.URL.Scheme) == 0 || len(via) >= p.MaxRedirects {
				return ErrTooManyRedirects
			}
			// A source URL may carry a credential in its query. Never disclose
			// it to the redirect target.
			request.Header.Del("Referer")
			_, err := p.ValidateURL(request.URL.String())
			return err
		},
	}
}

func (p Policy) dial(
	ctx context.Context,
	resolver IPResolver,
	dialer *net.Dialer,
	network string,
	address string,
) (net.Conn, error) {
	host, port, err := net.SplitHostPort(address)
	if err != nil {
		return nil, err
	}

	addresses, err := resolver.LookupNetIP(ctx, "ip", host)
	if err != nil {
		return nil, fmt.Errorf("%w: %w", ErrDestinationForbidden, err)
	}
	if len(addresses) == 0 {
		return nil, ErrDestinationForbidden
	}

	// Every answer must be public, not merely the one that gets dialled. A
	// mixed answer means the name is at least partly under someone else's
	// control, and which record is used is not this code's decision to retry.
	for _, candidate := range addresses {
		if !p.destinationAllowed(candidate) {
			return nil, ErrDestinationForbidden
		}
	}

	// Every candidate is validated, so any of them is safe to dial. Try them in
	// turn rather than pinning the first: the stock dialer resolves the name
	// itself and falls back across answers, and dialing one address would drop
	// that. A host whose first record is unreachable, which is the ordinary
	// shape of a partial IPv6 outage or a multi-A record with one bad host,
	// would otherwise fail outright.
	var errs error
	for _, candidate := range orderByFamily(addresses, network) {
		conn, dialErr := dialer.DialContext(ctx, network, net.JoinHostPort(candidate.String(), port))
		if dialErr == nil {
			return conn, nil
		}
		errs = errors.Join(errs, dialErr)
		// The overall deadline belongs to the caller. Once it is gone, further
		// attempts only burn time we no longer have.
		if ctx.Err() != nil {
			break
		}
	}
	if errs == nil {
		return nil, ErrDestinationForbidden
	}
	return nil, errs
}

// orderByFamily returns the validated addresses with the two families
// interleaved, keeping the resolver's order within each and leading with the
// family the resolver put first. Trying every IPv6 answer before the first IPv4
// one is what makes a broken IPv6 path look like a dead host.
//
// Addresses the transport cannot use for this network are dropped, so a tcp4
// dial does not spend an attempt on an IPv6 answer.
func orderByFamily(addresses []netip.Addr, network string) []netip.Addr {
	var v4, v6 []netip.Addr
	for _, a := range addresses {
		a = a.Unmap()
		switch {
		case a.Is4():
			if network != "tcp6" && network != "udp6" {
				v4 = append(v4, a)
			}
		default:
			if network != "tcp4" && network != "udp4" {
				v6 = append(v6, a)
			}
		}
	}

	first, second := v4, v6
	if len(addresses) > 0 && !addresses[0].Unmap().Is4() {
		first, second = v6, v4
	}

	ordered := make([]netip.Addr, 0, len(first)+len(second))
	for i := 0; i < len(first) || i < len(second); i++ {
		if i < len(first) {
			ordered = append(ordered, first[i])
		}
		if i < len(second) {
			ordered = append(ordered, second[i])
		}
	}
	return ordered
}

// ReadBounded reads at most limit bytes and reports ErrResponseTooLarge if the
// body is longer, rather than silently returning a truncated document.
func ReadBounded(body io.Reader, limit int64) ([]byte, error) {
	if limit <= 0 {
		return nil, ErrResponseTooLarge
	}
	data, err := io.ReadAll(io.LimitReader(body, limit+1))
	if err != nil {
		return nil, err
	}
	if int64(len(data)) > limit {
		return nil, ErrResponseTooLarge
	}
	return data, nil
}
