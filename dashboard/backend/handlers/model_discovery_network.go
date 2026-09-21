package handlers

import (
	"context"
	"errors"
	"fmt"
	"net"
	"net/http"
	"net/netip"
	"net/url"
	"strings"
	"time"

	modelcatalog "github.com/vllm-project/semantic-router/src/semantic-router/pkg/catalog"
)

const modelDiscoveryTimeout = 12 * time.Second

type modelDiscoveryTarget struct {
	url    string
	policy modelDiscoveryNetworkPolicy
}

type modelDiscoveryNetworkPolicy struct {
	allowPrivate bool
}

type modelDiscoveryNetworkPolicyKey struct{}

var modelDiscoveryBlockedPrefixes = []netip.Prefix{
	netip.MustParsePrefix("0.0.0.0/8"),
	netip.MustParsePrefix("100.64.0.0/10"),
	// Azure exposes host-agent services on this otherwise globally routed IP.
	netip.MustParsePrefix("168.63.129.16/32"),
	netip.MustParsePrefix("192.0.0.0/24"),
	netip.MustParsePrefix("192.0.2.0/24"),
	netip.MustParsePrefix("192.88.99.0/24"),
	netip.MustParsePrefix("198.18.0.0/15"),
	netip.MustParsePrefix("198.51.100.0/24"),
	netip.MustParsePrefix("203.0.113.0/24"),
	netip.MustParsePrefix("240.0.0.0/4"),
	// Translation and transition prefixes can otherwise tunnel an IPv4 target
	// past the address classification applied below.
	netip.MustParsePrefix("64:ff9b::/96"),
	netip.MustParsePrefix("64:ff9b:1::/48"),
	netip.MustParsePrefix("100::/64"),
	netip.MustParsePrefix("2001::/23"),
	netip.MustParsePrefix("2001:db8::/32"),
	netip.MustParsePrefix("2002::/16"),
	netip.MustParsePrefix("3fff::/20"),
	netip.MustParsePrefix("5f00::/16"),
	netip.MustParsePrefix("fec0::/10"),
}

func secureModelDiscoveryClient(client *http.Client) http.Client {
	if client == nil {
		client = &http.Client{Timeout: modelDiscoveryTimeout}
	}
	secured := *client
	if secured.Timeout <= 0 || secured.Timeout > modelDiscoveryTimeout {
		secured.Timeout = modelDiscoveryTimeout
	}
	// Discovery requests carry operator credentials. Do not attach cookies from
	// an injected client, even when the destination happens to share an origin.
	secured.Jar = nil
	secured.CheckRedirect = func(_ *http.Request, _ []*http.Request) error {
		return http.ErrUseLastResponse
	}

	switch transport := client.Transport.(type) {
	case nil:
		secured.Transport = secureModelDiscoveryTransport(http.DefaultTransport.(*http.Transport))
	case *http.Transport:
		secured.Transport = secureModelDiscoveryTransport(transport)
	default:
		// A wrapper can own its own proxy or dial path, so it cannot prove that the
		// per-request network policy is enforced. Fall back to the secured default;
		// callers that need custom roots or mTLS can pass an *http.Transport.
		secured.Transport = secureModelDiscoveryTransport(http.DefaultTransport.(*http.Transport))
	}
	return secured
}

func secureModelDiscoveryTransport(base *http.Transport) *http.Transport {
	transport := base.Clone()
	transport.Proxy = nil
	transport.DialTLS = nil
	transport.DialTLSContext = nil
	transport.DialContext = modelDiscoveryDialContext
	// Force a fresh validated DNS resolution and socket for every discovery.
	// A pooled connection could otherwise outlive a DNS change and bypass the
	// policy associated with a later request to the same host.
	transport.DisableKeepAlives = true
	return transport
}

func withModelDiscoveryNetworkPolicy(ctx context.Context, policy modelDiscoveryNetworkPolicy) context.Context {
	return context.WithValue(ctx, modelDiscoveryNetworkPolicyKey{}, policy)
}

func modelDiscoveryNetworkPolicyForProvider(parsed *url.URL, provider modelcatalog.ProviderDefinition) (modelDiscoveryNetworkPolicy, error) {
	var policy modelDiscoveryNetworkPolicy
	switch provider.Category {
	case "model_api":
		catalogURL, err := url.Parse(provider.DefaultBaseURL)
		if err != nil || catalogURL.Host == "" || !sameModelDiscoveryOrigin(parsed, catalogURL) {
			return modelDiscoveryNetworkPolicy{}, errors.New("cloud provider discovery must use the built-in provider origin")
		}
	case "private_runtime", "start_here":
		policy.allowPrivate = true
	default:
		return modelDiscoveryNetworkPolicy{}, errors.New("this provider does not declare a supported discovery network policy")
	}
	if literal, err := netip.ParseAddr(parsed.Hostname()); err == nil && !modelDiscoveryAddressAllowed(literal, policy) {
		return modelDiscoveryNetworkPolicy{}, errors.New("the provider URL targets a restricted network address")
	}
	return policy, nil
}

func sameModelDiscoveryOrigin(left, right *url.URL) bool {
	return strings.EqualFold(left.Scheme, right.Scheme) &&
		strings.EqualFold(left.Hostname(), right.Hostname()) &&
		modelDiscoveryPort(left) == modelDiscoveryPort(right)
}

func modelDiscoveryPort(value *url.URL) string {
	if value.Port() != "" {
		return value.Port()
	}
	if strings.EqualFold(value.Scheme, "https") {
		return "443"
	}
	return "80"
}

func modelDiscoveryDialContext(ctx context.Context, network, address string) (net.Conn, error) {
	policy, ok := ctx.Value(modelDiscoveryNetworkPolicyKey{}).(modelDiscoveryNetworkPolicy)
	if !ok {
		return nil, errors.New("model discovery network policy is missing")
	}
	host, port, err := net.SplitHostPort(address)
	if err != nil {
		return nil, fmt.Errorf("invalid model discovery address: %w", err)
	}
	lookupNetwork := "ip"
	if strings.HasSuffix(network, "4") {
		lookupNetwork = "ip4"
	} else if strings.HasSuffix(network, "6") {
		lookupNetwork = "ip6"
	}
	addresses, err := net.DefaultResolver.LookupNetIP(ctx, lookupNetwork, host)
	if err != nil || len(addresses) == 0 {
		return nil, errors.New("model discovery host could not be resolved")
	}
	for _, resolved := range addresses {
		if !modelDiscoveryAddressAllowed(resolved, policy) {
			return nil, errors.New("model discovery host resolves to a restricted network address")
		}
	}

	dialer := &net.Dialer{Timeout: modelDiscoveryTimeout, KeepAlive: 30 * time.Second}
	var lastError error
	for _, resolved := range addresses {
		connection, dialErr := dialer.DialContext(ctx, network, net.JoinHostPort(resolved.String(), port))
		if dialErr == nil {
			return connection, nil
		}
		lastError = dialErr
	}
	return nil, lastError
}

func modelDiscoveryAddressAllowed(address netip.Addr, policy modelDiscoveryNetworkPolicy) bool {
	if !address.IsValid() {
		return false
	}
	address = address.Unmap()
	if address.IsUnspecified() || address.IsMulticast() || address.IsLinkLocalUnicast() || address.IsLinkLocalMulticast() {
		return false
	}
	for _, prefix := range modelDiscoveryBlockedPrefixes {
		if prefix.Contains(address) {
			return false
		}
	}
	if address.IsLoopback() || address.IsPrivate() {
		return policy.allowPrivate
	}
	return address.IsGlobalUnicast()
}
