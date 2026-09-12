package safefetch

import (
	"net/netip"
	"testing"
)

// Every class named in the acceptance criteria, in both address families, plus
// the notations that let one be written to look like another.
func TestIsPublicAddrRejectsNonPublicClasses(t *testing.T) {
	denied := map[string][]string{
		"loopback":           {"127.0.0.1", "127.1.2.3", "::1"},
		"unspecified":        {"0.0.0.0", "::"},
		"private":            {"10.0.0.1", "172.16.0.1", "172.31.255.254", "192.168.1.1", "fd00::1", "fc00::1"},
		"link-local":         {"169.254.1.1", "fe80::1"},
		"cloud metadata":     {"169.254.169.254"},
		"carrier-grade NAT":  {"100.64.0.1"},
		"multicast":          {"224.0.0.1", "239.255.255.255", "ff02::1"},
		"reserved":           {"240.0.0.1", "255.255.255.255"},
		"documentation":      {"192.0.2.1", "198.51.100.1", "203.0.113.1", "2001:db8::1"},
		"benchmarking":       {"198.18.0.1"},
		"IETF protocol":      {"192.0.0.1"},
		"site-local v6":      {"fec0::1"},
		"6to4":               {"2002::1"},
		"Teredo":             {"2001::1"},
		"NAT64":              {"64:ff9b::7f00:1"},
		"IPv4-mapped v6":     {"::ffff:127.0.0.1", "::ffff:10.0.0.1", "::ffff:169.254.169.254"},
		"IPv4-compatible v6": {"::127.0.0.1"},
	}

	for class, addresses := range denied {
		for _, raw := range addresses {
			address, err := netip.ParseAddr(raw)
			if err != nil {
				t.Fatalf("%s: %q is not parseable: %v", class, raw, err)
			}
			if IsPublicAddr(address) {
				t.Errorf("%s: %q was accepted as public", class, raw)
			}
		}
	}
}

func TestIsPublicAddrAcceptsPublicAddresses(t *testing.T) {
	for _, raw := range []string{"8.8.8.8", "1.1.1.1", "93.184.216.34", "2606:4700:4700::1111"} {
		address, err := netip.ParseAddr(raw)
		if err != nil {
			t.Fatalf("%q is not parseable: %v", raw, err)
		}
		if !IsPublicAddr(address) {
			t.Errorf("%q was refused, want accepted", raw)
		}
	}
}

func TestIsPublicAddrRejectsZeroValue(t *testing.T) {
	if IsPublicAddr(netip.Addr{}) {
		t.Error("the zero Addr was accepted as public")
	}
}
