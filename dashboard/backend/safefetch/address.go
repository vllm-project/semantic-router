// Package safefetch is the one outbound-fetch policy for caller-supplied URLs.
//
// Any handler that dials a destination the caller chose goes through this
// package. The policy is deny-by-default on the resolved address, not on the
// hostname: a name check alone is decided before DNS, so it cannot see where
// the connection actually goes.
package safefetch

import "net/netip"

// nonPublicNetworks are the ranges a public-web fetch must never reach. It
// covers loopback, private, carrier-grade NAT, link-local (including the cloud
// metadata address 169.254.169.254), documentation, benchmarking, multicast,
// and the IPv6 equivalents plus the translation ranges that embed an IPv4
// destination.
var nonPublicNetworks = mustPrefixes(
	"0.0.0.0/8", "10.0.0.0/8", "100.64.0.0/10", "127.0.0.0/8",
	"169.254.0.0/16", "172.16.0.0/12", "192.0.0.0/24", "192.0.2.0/24",
	"192.88.99.0/24", "192.168.0.0/16", "198.18.0.0/15", "198.51.100.0/24",
	"203.0.113.0/24", "224.0.0.0/4", "240.0.0.0/4",
	"::/96", "::1/128", "64:ff9b::/96", "64:ff9b:1::/48", "100::/64",
	"2001::/32", "2001:2::/48", "2001:10::/28", "2001:20::/28",
	"2001:db8::/32", "2002::/16", "fc00::/7", "fec0::/10", "fe80::/10", "ff00::/8",
)

// IsPublicAddr reports whether address is routable on the public internet.
//
// An IPv4-mapped IPv6 address is unmapped first, so ::ffff:127.0.0.1 is judged
// as 127.0.0.1 rather than passing as a global unicast v6 address.
func IsPublicAddr(address netip.Addr) bool {
	if !address.IsValid() {
		return false
	}
	address = address.Unmap()
	if !address.IsGlobalUnicast() {
		return false
	}
	for _, prefix := range nonPublicNetworks {
		if prefix.Contains(address) {
			return false
		}
	}
	return true
}

func mustPrefixes(values ...string) []netip.Prefix {
	result := make([]netip.Prefix, 0, len(values))
	for _, value := range values {
		result = append(result, netip.MustParsePrefix(value))
	}
	return result
}
