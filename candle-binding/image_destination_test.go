package candle_binding

import (
	"net/netip"
	"testing"
)

func TestCandleImageDestinationAppliesIPv6PublicRoutingPolicy(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name    string
		address string
		want    bool
	}{
		{name: "cloudflare public dns", address: "2606:4700:4700::1111", want: true},
		{name: "discard only", address: "100::1"},
		{name: "dummy prefix", address: "100:0:0:1::1"},
		{name: "IETF protocol assignments", address: "2001::1"},
		{name: "documentation", address: "2001:db8::1"},
		{name: "6to4 tunnel", address: "2002:c000:204::1"},
		{name: "documentation version 2", address: "3fff::1"},
		{name: "segment routing SIDs", address: "5f00::1"},
		{name: "deprecated site local", address: "fec0::1"},
		{name: "outside allocated global unicast", address: "4000::1"},
	}

	for _, test := range tests {
		test := test
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			got := isPublicCandleImageDestination(netip.MustParseAddr(test.address))
			if got != test.want {
				t.Fatalf("isPublicCandleImageDestination(%s) = %v, want %v", test.address, got, test.want)
			}
		})
	}
}

func TestCandleImageDestinationRejectsNonPublicTranslatedIPv4(t *testing.T) {
	t.Parallel()

	for _, ipv4 := range []string{
		"0.0.0.0",
		"10.0.0.1",
		"127.0.0.1",
		"169.254.169.254",
		"198.18.0.1",
	} {
		ipv4 := netip.MustParseAddr(ipv4)
		for _, translated := range []netip.Addr{testRFC6052Address(rfc6052WellKnownTranslationPrefix, ipv4)} {
			if isPublicCandleImageDestination(translated) {
				t.Fatalf("translated non-public destination %s (IPv4 %s) must be blocked", translated, ipv4)
			}
		}
	}
}

func TestCandleImageDestinationAllowsPublicTranslatedIPv4(t *testing.T) {
	t.Parallel()

	ipv4 := netip.MustParseAddr("1.1.1.1")
	for _, translated := range []netip.Addr{testRFC6052Address(rfc6052WellKnownTranslationPrefix, ipv4)} {
		if !isPublicCandleImageDestination(translated) {
			t.Fatalf("translated public destination %s must be allowed", translated)
		}
	}
}

func TestCandleImageDestinationRejectsEntireRFC8215LocalUseBlock(t *testing.T) {
	t.Parallel()

	for _, address := range []string{
		"64:ff9b:1::1",
		"64:ff9b:1::1.1.1.1",
		"64:ff9b:1:abcd:0:5431:101:101",
	} {
		if isPublicCandleImageDestination(netip.MustParseAddr(address)) {
			t.Fatalf("RFC 8215 local-use destination %s must be blocked", address)
		}
	}
}

func testRFC6052Address(prefix netip.Prefix, ipv4 netip.Addr) netip.Addr {
	bytes := prefix.Addr().As16()
	v4 := ipv4.As4()
	switch prefix.Bits() {
	case 48:
		bytes[6], bytes[7] = v4[0], v4[1]
		bytes[8] = 0
		bytes[9], bytes[10] = v4[2], v4[3]
	case 96:
		copy(bytes[12:], v4[:])
	default:
		panic("test helper supports only /48 and /96 translation prefixes")
	}
	return netip.AddrFrom16(bytes)
}
