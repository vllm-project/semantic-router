package handlers

import (
	"net/netip"
	"strings"
)

var specialUseIPv4Prefixes = []netip.Prefix{
	netip.MustParsePrefix("0.0.0.0/8"),
	netip.MustParsePrefix("10.0.0.0/8"),
	netip.MustParsePrefix("100.64.0.0/10"),
	netip.MustParsePrefix("127.0.0.0/8"),
	netip.MustParsePrefix("169.254.0.0/16"),
	netip.MustParsePrefix("172.16.0.0/12"),
	netip.MustParsePrefix("192.0.0.0/24"),
	netip.MustParsePrefix("192.0.2.0/24"),
	netip.MustParsePrefix("192.31.196.0/24"),
	netip.MustParsePrefix("192.52.193.0/24"),
	netip.MustParsePrefix("192.88.99.0/24"),
	netip.MustParsePrefix("192.168.0.0/16"),
	netip.MustParsePrefix("192.175.48.0/24"),
	netip.MustParsePrefix("198.18.0.0/15"),
	netip.MustParsePrefix("198.51.100.0/24"),
	netip.MustParsePrefix("203.0.113.0/24"),
	netip.MustParsePrefix("224.0.0.0/4"),
	netip.MustParsePrefix("240.0.0.0/4"),
}

var specialUseIPv6Prefixes = []netip.Prefix{
	netip.MustParsePrefix("::/96"),
	netip.MustParsePrefix("64:ff9b::/96"),
	netip.MustParsePrefix("64:ff9b:1::/48"),
	netip.MustParsePrefix("100::/64"),
	netip.MustParsePrefix("2001::/23"),
	netip.MustParsePrefix("2001:db8::/32"),
	netip.MustParsePrefix("2002::/16"),
	netip.MustParsePrefix("2620:4f:8000::/48"),
	netip.MustParsePrefix("3fff::/20"),
	netip.MustParsePrefix("5f00::/16"),
	netip.MustParsePrefix("fc00::/7"),
	netip.MustParsePrefix("fe80::/10"),
	netip.MustParsePrefix("fec0::/10"),
	netip.MustParsePrefix("ff00::/8"),
}

var allocatedGlobalUnicastIPv6Prefix = netip.MustParsePrefix("2000::/3")

func normalizeOutboundHostname(host string) string {
	return strings.TrimSuffix(strings.ToLower(strings.TrimSpace(host)), ".")
}

func isBlockedOutboundHostname(host string) bool {
	return host == "localhost" || strings.HasSuffix(host, ".localhost")
}

// isAmbiguousIPv4Hostname rejects legacy numeric spellings such as 127.1,
// 0177.0.0.1, and 0x7f000001. Different URL parsers, resolvers, and proxies
// disagree on their meaning; allowing DNS resolution would make the policy
// vulnerable to a later component interpreting one as a private IPv4 literal.
// Canonical dotted-decimal addresses have already been handled by ParseAddr.
func isAmbiguousIPv4Hostname(host string) bool {
	parts := strings.Split(host, ".")
	if len(parts) == 0 || len(parts) > 4 {
		return false
	}
	for _, part := range parts {
		if part == "" {
			return false
		}
		digits := part
		base := 10
		if strings.HasPrefix(strings.ToLower(part), "0x") {
			digits = part[2:]
			base = 16
		}
		if digits == "" {
			return false
		}
		for _, value := range digits {
			if base == 10 && !strings.ContainsRune("0123456789", value) {
				return false
			}
			if base == 16 && !strings.ContainsRune("0123456789abcdefABCDEF", value) {
				return false
			}
		}
	}
	return true
}

func isPublicOutboundIP(address netip.Addr) bool {
	if !address.IsValid() || address.Zone() != "" {
		return false
	}
	address = address.Unmap()
	if !address.IsGlobalUnicast() {
		return false
	}
	if address.Is6() && !allocatedGlobalUnicastIPv6Prefix.Contains(address) {
		return false
	}

	prefixes := specialUseIPv6Prefixes
	if address.Is4() {
		prefixes = specialUseIPv4Prefixes
	}
	for _, prefix := range prefixes {
		if prefix.Contains(address) {
			return false
		}
	}
	return true
}
