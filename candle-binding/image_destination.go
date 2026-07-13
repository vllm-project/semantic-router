package candle_binding

import (
	"context"
	"net/netip"
)

var publicCandleImageDestinationIPv6Prefix = netip.MustParsePrefix("2000::/3")

var blockedCandleImageDestinationPrefixes = [...]netip.Prefix{
	netip.MustParsePrefix("0.0.0.0/8"),
	netip.MustParsePrefix("10.0.0.0/8"),
	netip.MustParsePrefix("100.64.0.0/10"),
	netip.MustParsePrefix("127.0.0.0/8"),
	netip.MustParsePrefix("169.254.0.0/16"),
	netip.MustParsePrefix("172.16.0.0/12"),
	netip.MustParsePrefix("192.0.0.0/24"),
	netip.MustParsePrefix("192.0.2.0/24"),
	netip.MustParsePrefix("192.168.0.0/16"),
	netip.MustParsePrefix("198.18.0.0/15"),
	netip.MustParsePrefix("198.51.100.0/24"),
	netip.MustParsePrefix("203.0.113.0/24"),
	netip.MustParsePrefix("224.0.0.0/4"),
	netip.MustParsePrefix("240.0.0.0/4"),
	netip.MustParsePrefix("::/128"),
	netip.MustParsePrefix("::1/128"),
	netip.MustParsePrefix("100::/64"),
	netip.MustParsePrefix("100:0:0:1::/64"),
	netip.MustParsePrefix("2001::/23"),
	netip.MustParsePrefix("2001:db8::/32"),
	netip.MustParsePrefix("2002::/16"),
	netip.MustParsePrefix("3fff::/20"),
	netip.MustParsePrefix("5f00::/16"),
	netip.MustParsePrefix("fc00::/7"),
	netip.MustParsePrefix("fe80::/10"),
	netip.MustParsePrefix("fec0::/10"),
	netip.MustParsePrefix("ff00::/8"),
	// RFC 8215 reserves this entire block for domain-local, technology-agnostic
	// translation and explicitly says applications cannot assume where an IPv4
	// address is embedded. It is not globally reachable, so a public-only fetch
	// policy must reject the block rather than infer one layout.
	netip.MustParsePrefix("64:ff9b:1::/48"),
}

func isAllowedDiscoveredPref64(prefix netip.Prefix) bool {
	if _, ok := rfc6052LayoutForPrefix(prefix); !ok {
		return false
	}
	// The RFC 6052 well-known prefix is globally usable translation space even
	// though it sits outside the ordinary 2000::/3 global-unicast allocation.
	if prefix == rfc6052WellKnownTranslationPrefix {
		return !pref64OverlapsBlockedImageDestination(prefix)
	}
	if !isPublicPref64Address(prefix.Addr()) {
		return false
	}
	return !pref64OverlapsBlockedImageDestination(prefix)
}

func isPublicPref64Address(address netip.Addr) bool {
	return isPublicCandleNativeIPv6Destination(address)
}

func pref64OverlapsBlockedImageDestination(prefix netip.Prefix) bool {
	for _, blocked := range blockedCandleImageDestinationPrefixes {
		if !blocked.Addr().Is6() {
			continue
		}
		if blocked.Contains(prefix.Addr()) || prefix.Contains(blocked.Addr()) {
			return true
		}
	}
	return false
}

func isPublicCandleImageDestination(address netip.Addr) bool {
	if !address.IsValid() || address.Zone() != "" {
		return false
	}
	address = address.Unmap()
	if embedded, ok := embeddedTranslationIPv4(address); ok {
		return isPublicCandleImageDestination(embedded)
	}
	if address.Is6() {
		return isPublicCandleNativeIPv6Destination(address)
	}
	if !address.IsGlobalUnicast() || address.IsPrivate() ||
		address.IsLoopback() || address.IsLinkLocalUnicast() {
		return false
	}
	for _, prefix := range blockedCandleImageDestinationPrefixes {
		if prefix.Contains(address) {
			return false
		}
	}
	return true
}

func isPublicCandleNativeIPv6Destination(address netip.Addr) bool {
	if !address.IsValid() || address.Zone() != "" || !address.Is6() || address.Is4In6() {
		return false
	}
	// netip.Addr.IsGlobalUnicast reports several IANA special-purpose ranges as
	// global unicast. Restrict ordinary destinations to currently allocated
	// global-unicast space, then explicitly subtract non-public registrations.
	if !publicCandleImageDestinationIPv6Prefix.Contains(address) ||
		!address.IsGlobalUnicast() || address.IsPrivate() ||
		address.IsLoopback() || address.IsLinkLocalUnicast() {
		return false
	}
	for _, prefix := range blockedCandleImageDestinationPrefixes {
		if prefix.Contains(address) {
			return false
		}
	}
	return true
}

func isPublicCandleImageDestinationWithPrefixes(address netip.Addr, prefixes []netip.Prefix) bool {
	if !isPublicCandleImageDestination(address) {
		return false
	}
	address = address.Unmap()
	if _, ok := embeddedTranslationIPv4(address); ok {
		return true
	}

	var embedded netip.Addr
	matches := 0
	for _, prefix := range prefixes {
		if !isAllowedDiscoveredPref64(prefix) {
			return false
		}
		if !prefix.Contains(address) {
			continue
		}
		decoded, ok := extractRFC6052IPv4(address, prefix)
		if !ok {
			return false
		}
		matches++
		if matches > 1 {
			return false
		}
		embedded = decoded
	}
	if matches == 1 {
		return isPublicCandleImageDestination(embedded)
	}
	return true
}

func candleImageDestinationsArePublic(
	ctx context.Context,
	addresses []netip.Addr,
	cache *pref64DiscoveryCache,
) bool {
	if len(addresses) == 0 {
		return false
	}

	needsDiscovery := false
	for _, address := range addresses {
		if !isPublicCandleImageDestination(address) {
			return false
		}
		unmapped := address.Unmap()
		if unmapped.Is6() {
			if _, wellKnown := embeddedTranslationIPv4(unmapped); !wellKnown {
				needsDiscovery = true
			}
		}
	}
	if !needsDiscovery {
		return true
	}

	prefixes, err := cache.prefixesFor(ctx)
	if err != nil {
		return false
	}
	for _, address := range addresses {
		if !isPublicCandleImageDestinationWithPrefixes(address, prefixes) {
			return false
		}
	}
	return true
}
