package onnx_binding

import (
	"net/netip"
	"testing"
)

func TestONNXImageDestinationAppliesIPv6PublicRoutingPolicy(t *testing.T) {
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
			got := isPublicImageDestination(netip.MustParseAddr(test.address))
			if got != test.want {
				t.Fatalf("isPublicImageDestination(%s) = %v, want %v", test.address, got, test.want)
			}
		})
	}
}
