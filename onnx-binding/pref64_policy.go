package onnx_binding

import (
	"context"
	"errors"
	"fmt"
	"net"
	"net/netip"
	"sync"
	"time"
)

const (
	rfc7050DiscoveryName      = "ipv4only.arpa."
	pref64DiscoverySuccessTTL = 30 * time.Second
	pref64DiscoveryFailureTTL = 5 * time.Second
	pref64DiscoveryTimeout    = 2 * time.Second
)

type rfc6052Layout struct {
	prefixBits      int
	ipv4ByteIndexes [4]uint8
}

var (
	rfc6052WellKnownTranslationPrefix = netip.MustParsePrefix("64:ff9b::/96")
	rfc7050IPv4OnlyAddresses          = [...]netip.Addr{
		netip.MustParseAddr("192.0.0.170"),
		netip.MustParseAddr("192.0.0.171"),
	}
	rfc6052Layouts = [...]rfc6052Layout{
		{prefixBits: 32, ipv4ByteIndexes: [4]uint8{4, 5, 6, 7}},
		{prefixBits: 40, ipv4ByteIndexes: [4]uint8{5, 6, 7, 9}},
		{prefixBits: 48, ipv4ByteIndexes: [4]uint8{6, 7, 9, 10}},
		{prefixBits: 56, ipv4ByteIndexes: [4]uint8{7, 9, 10, 11}},
		{prefixBits: 64, ipv4ByteIndexes: [4]uint8{9, 10, 11, 12}},
		{prefixBits: 96, ipv4ByteIndexes: [4]uint8{12, 13, 14, 15}},
	}
)

type imageAddressResolver interface {
	LookupNetIP(context.Context, string, string) ([]netip.Addr, error)
}

// pref64DiscoveryCache keeps RFC 7050 discovery bounded and coalesces
// concurrent lookups. net.Resolver does not expose DNS TTLs, so the cache uses
// deliberately short lifetimes to track network changes without doing two DNS
// queries for every public IPv6 image request.
type pref64DiscoveryCache struct {
	mu            sync.Mutex
	resolver      imageAddressResolver
	now           func() time.Time
	successTTL    time.Duration
	failureTTL    time.Duration
	lookupTimeout time.Duration
	expires       time.Time
	prefixes      []netip.Prefix
	err           error
	ready         bool
	inFlight      chan struct{}
}

var onnxImagePref64Cache = newPref64DiscoveryCache(
	net.DefaultResolver,
	pref64DiscoverySuccessTTL,
	pref64DiscoveryFailureTTL,
	pref64DiscoveryTimeout,
)

func newPref64DiscoveryCache(
	resolver imageAddressResolver,
	successTTL time.Duration,
	failureTTL time.Duration,
	lookupTimeout time.Duration,
) *pref64DiscoveryCache {
	return &pref64DiscoveryCache{
		resolver:      resolver,
		now:           time.Now,
		successTTL:    successTTL,
		failureTTL:    failureTTL,
		lookupTimeout: lookupTimeout,
	}
}

func (cache *pref64DiscoveryCache) prefixesFor(ctx context.Context) ([]netip.Prefix, error) {
	if cache == nil || cache.resolver == nil {
		return nil, errors.New("PREF64 discovery is unavailable")
	}

	for {
		cache.mu.Lock()
		if cache.ready && cache.now().Before(cache.expires) {
			prefixes := append([]netip.Prefix(nil), cache.prefixes...)
			err := cache.err
			cache.mu.Unlock()
			return prefixes, err
		}
		if wait := cache.inFlight; wait != nil {
			cache.mu.Unlock()
			select {
			case <-wait:
				continue
			case <-ctx.Done():
				return nil, ctx.Err()
			}
		}

		wait := make(chan struct{})
		cache.inFlight = wait
		cache.mu.Unlock()

		discoveryCtx, cancel := context.WithTimeout(ctx, cache.lookupTimeout)
		prefixes, err := discoverRFC7050Prefixes(discoveryCtx, cache.resolver)
		cancel()

		cache.mu.Lock()
		cache.prefixes = append(cache.prefixes[:0], prefixes...)
		cache.err = err
		cache.ready = true
		ttl := cache.successTTL
		if err != nil {
			ttl = cache.failureTTL
		}
		cache.expires = cache.now().Add(ttl)
		cache.inFlight = nil
		close(wait)
		cache.mu.Unlock()
		return append([]netip.Prefix(nil), prefixes...), err
	}
}

func discoverRFC7050Prefixes(ctx context.Context, resolver imageAddressResolver) ([]netip.Prefix, error) {
	addresses, err := resolver.LookupNetIP(ctx, "ip6", rfc7050DiscoveryName)
	if err == nil && len(addresses) != 0 {
		return inferRFC7050Prefixes(addresses)
	}
	if err != nil && !isDNSNotFound(err) {
		return nil, fmt.Errorf("resolve RFC 7050 IPv6 discovery name: %w", err)
	}

	// A positive A-only result proves that the active resolver can resolve the
	// RFC 7050 name but is not synthesizing AAAA records. A lookup failure is
	// not equivalent to "no DNS64" and therefore fails closed.
	ipv4Addresses, ipv4Err := resolver.LookupNetIP(ctx, "ip4", rfc7050DiscoveryName)
	if ipv4Err != nil {
		return nil, fmt.Errorf("verify RFC 7050 IPv4 discovery name: %w", ipv4Err)
	}
	if len(ipv4Addresses) == 0 {
		return nil, errors.New("RFC 7050 discovery name resolved to no addresses")
	}
	for _, address := range ipv4Addresses {
		address = address.Unmap()
		if !address.Is4() || !isRFC7050IPv4OnlyAddress(address) {
			return nil, errors.New("RFC 7050 discovery name returned an unexpected IPv4 address")
		}
	}
	return nil, nil
}

func inferRFC7050Prefixes(addresses []netip.Addr) ([]netip.Prefix, error) {
	if len(addresses) == 0 {
		return nil, errors.New("RFC 7050 discovery returned no synthesized IPv6 addresses")
	}

	seen := make(map[netip.Prefix]struct{})
	prefixes := make([]netip.Prefix, 0, len(addresses))
	for _, address := range addresses {
		if !address.Is6() || address.Is4In6() || address.Zone() != "" {
			return nil, errors.New("RFC 7050 discovery returned a non-IPv6 address")
		}

		candidates := make([]netip.Prefix, 0, 1)
		for _, layout := range rfc6052Layouts {
			prefix := netip.PrefixFrom(address, layout.prefixBits).Masked()
			embedded, ok := extractRFC6052IPv4(address, prefix)
			if ok && isRFC7050IPv4OnlyAddress(embedded) && isAllowedDiscoveredPref64(prefix) {
				candidates = append(candidates, prefix)
			}
		}
		if len(candidates) != 1 {
			return nil, fmt.Errorf("RFC 7050 address has %d valid PREF64 interpretations", len(candidates))
		}
		if _, ok := seen[candidates[0]]; !ok {
			seen[candidates[0]] = struct{}{}
			prefixes = append(prefixes, candidates[0])
		}
	}
	return prefixes, nil
}

func isDNSNotFound(err error) bool {
	var dnsErr *net.DNSError
	return errors.As(err, &dnsErr) && dnsErr.IsNotFound
}

func isRFC7050IPv4OnlyAddress(address netip.Addr) bool {
	for _, known := range rfc7050IPv4OnlyAddresses {
		if address == known {
			return true
		}
	}
	return false
}

func rfc6052LayoutForPrefix(prefix netip.Prefix) (rfc6052Layout, bool) {
	if !prefix.IsValid() || prefix != prefix.Masked() || !prefix.Addr().Is6() {
		return rfc6052Layout{}, false
	}
	for _, layout := range rfc6052Layouts {
		if prefix.Bits() == layout.prefixBits {
			return layout, true
		}
	}
	return rfc6052Layout{}, false
}

func rfc6052LayoutForAddress(address netip.Addr, prefix netip.Prefix) (rfc6052Layout, bool) {
	layout, ok := rfc6052LayoutForPrefix(prefix)
	if !ok || !address.Is6() || address.Is4In6() || address.Zone() != "" || !prefix.Contains(address) {
		return rfc6052Layout{}, false
	}
	return layout, true
}

// embeddedTranslationIPv4 recognizes the fixed RFC 6052 well-known prefix.
// Network-specific prefixes are accepted only after RFC 7050 discovery.
func embeddedTranslationIPv4(address netip.Addr) (netip.Addr, bool) {
	return extractRFC6052IPv4(address, rfc6052WellKnownTranslationPrefix)
}

// extractRFC6052IPv4 implements all six standard layouts from RFC 6052. The
// reserved u octet (bits 64-71) must be zero, including when it is part of a
// /96 prefix. The suffix is intentionally ignored because translators are
// required to derive the IPv4 destination from the embedded 32 bits.
func extractRFC6052IPv4(address netip.Addr, prefix netip.Prefix) (netip.Addr, bool) {
	layout, ok := rfc6052LayoutForAddress(address, prefix)
	if !ok {
		return netip.Addr{}, false
	}
	bytes := address.As16()
	if bytes[8] != 0 {
		return netip.Addr{}, false
	}
	indexes := layout.ipv4ByteIndexes
	ipv4 := [4]byte{
		bytes[indexes[0]],
		bytes[indexes[1]],
		bytes[indexes[2]],
		bytes[indexes[3]],
	}
	return netip.AddrFrom4(ipv4), true
}
