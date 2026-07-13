package onnx_binding

import (
	"context"
	"errors"
	"net"
	"net/netip"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

type onnxImageResolverFunc func(context.Context, string, string) ([]netip.Addr, error)

func (resolve onnxImageResolverFunc) LookupNetIP(
	ctx context.Context,
	network string,
	host string,
) ([]netip.Addr, error) {
	return resolve(ctx, network, host)
}

func TestONNXRFC6052ExtractionSupportsEveryStandardPrefixLength(t *testing.T) {
	t.Parallel()

	ipv4 := netip.MustParseAddr("203.0.113.9")
	for _, prefix := range onnxRFC6052TestPrefixes() {
		prefix := prefix
		t.Run(prefix.String(), func(t *testing.T) {
			t.Parallel()
			translated := synthesizeONNXRFC6052Address(prefix, ipv4)
			if prefix.Bits() != 96 {
				withSuffix := translated.As16()
				withSuffix[15] = 0x5a
				translated = netip.AddrFrom16(withSuffix)
			}
			decoded, ok := extractRFC6052IPv4(translated, prefix)
			if !ok || decoded != ipv4 {
				t.Fatalf("decode %s with %s: got %s, ok=%v", translated, prefix, decoded, ok)
			}
		})
	}

	prefix := netip.MustParsePrefix("2001:4860:1234:5678::/64")
	malformed := synthesizeONNXRFC6052Address(prefix, ipv4).As16()
	malformed[8] = 1
	if _, ok := extractRFC6052IPv4(netip.AddrFrom16(malformed), prefix); ok {
		t.Fatal("non-zero RFC 6052 u octet must be rejected")
	}
	if _, ok := extractRFC6052IPv4(
		netip.MustParseAddr("2001:4860:1234:5678::1"),
		netip.MustParsePrefix("2001:4860:1234:5678::/72"),
	); ok {
		t.Fatal("non-standard RFC 6052 prefix length must be rejected")
	}
	if isPublicImageDestination(netip.MustParseAddr("2606:4700:4700::1111%en0")) {
		t.Fatal("scoped IPv6 destinations must be rejected")
	}
}

func TestONNXRFC7050InfersEveryStandardPrefixLength(t *testing.T) {
	t.Parallel()

	for _, prefix := range append(
		[]netip.Prefix{rfc6052WellKnownTranslationPrefix},
		onnxRFC6052TestPrefixes()...,
	) {
		prefix := prefix
		t.Run(prefix.String(), func(t *testing.T) {
			t.Parallel()
			addresses := []netip.Addr{
				synthesizeONNXRFC6052Address(prefix, rfc7050IPv4OnlyAddresses[0]),
				synthesizeONNXRFC6052Address(prefix, rfc7050IPv4OnlyAddresses[1]),
			}
			prefixes, err := inferRFC7050Prefixes(addresses)
			if err != nil {
				t.Fatalf("infer %s: %v", prefix, err)
			}
			if len(prefixes) != 1 || prefixes[0] != prefix {
				t.Fatalf("infer %s: got %v", prefix, prefixes)
			}
		})
	}
}

func TestONNXNetworkSpecificPref64AppliesPublicIPv4Policy(t *testing.T) {
	t.Parallel()

	for _, prefix := range onnxRFC6052TestPrefixes() {
		for _, ipv4 := range []string{"10.0.0.1", "127.0.0.1", "169.254.169.254"} {
			translated := synthesizeONNXRFC6052Address(prefix, netip.MustParseAddr(ipv4))
			if isPublicImageDestinationWithPrefixes(translated, []netip.Prefix{prefix}) {
				t.Fatalf("%s translation of non-public %s must be blocked", prefix, ipv4)
			}
		}
		translated := synthesizeONNXRFC6052Address(prefix, netip.MustParseAddr("1.1.1.1"))
		if !isPublicImageDestinationWithPrefixes(translated, []netip.Prefix{prefix}) {
			t.Fatalf("%s translation of public IPv4 must be allowed", prefix)
		}
	}
}

func TestONNXRFC7050DiscoveryNoDNS64AndFailures(t *testing.T) {
	t.Parallel()

	noDNS64 := onnxImageResolverFunc(func(_ context.Context, network, host string) ([]netip.Addr, error) {
		if host != rfc7050DiscoveryName {
			t.Fatalf("unexpected discovery host %q", host)
		}
		if network == "ip6" {
			return nil, &net.DNSError{IsNotFound: true}
		}
		return append([]netip.Addr(nil), rfc7050IPv4OnlyAddresses[:]...), nil
	})
	prefixes, err := discoverRFC7050Prefixes(context.Background(), noDNS64)
	if err != nil || len(prefixes) != 0 {
		t.Fatalf("A-only RFC 7050 response must prove no DNS64, prefixes=%v err=%v", prefixes, err)
	}
	noDNS64Cache := newPref64DiscoveryCache(noDNS64, time.Minute, time.Second, time.Second)
	if !imageDestinationsArePublic(
		context.Background(),
		[]netip.Addr{netip.MustParseAddr("2606:4700:4700::1111")},
		noDNS64Cache,
	) {
		t.Fatal("native public IPv6 must remain operable when RFC 7050 proves there is no DNS64")
	}

	discoveryFailure := errors.New("resolver unavailable")
	failing := onnxImageResolverFunc(func(context.Context, string, string) ([]netip.Addr, error) {
		return nil, discoveryFailure
	})
	if _, err := discoverRFC7050Prefixes(context.Background(), failing); err == nil {
		t.Fatal("discovery failure must not be treated as no DNS64")
	}

	invalidA := onnxImageResolverFunc(func(_ context.Context, network, _ string) ([]netip.Addr, error) {
		if network == "ip6" {
			return nil, &net.DNSError{IsNotFound: true}
		}
		return []netip.Addr{netip.MustParseAddr("198.51.100.1")}, nil
	})
	if _, err := discoverRFC7050Prefixes(context.Background(), invalidA); err == nil {
		t.Fatal("unexpected A response must fail RFC 7050 discovery")
	}
}

func TestONNXRFC7050RejectsAmbiguousAndLocalUsePrefixes(t *testing.T) {
	t.Parallel()

	ambiguousBytes := netip.MustParseAddr("2001:4860::").As16()
	wka := rfc7050IPv4OnlyAddresses[0].As4()
	copy(ambiguousBytes[4:8], wka[:])
	copy(ambiguousBytes[12:16], wka[:])
	if _, err := inferRFC7050Prefixes([]netip.Addr{netip.AddrFrom16(ambiguousBytes)}); err == nil {
		t.Fatal("ambiguous RFC 7050 address must be rejected")
	}

	localUse := netip.MustParsePrefix("64:ff9b:1::/48")
	if _, err := inferRFC7050Prefixes([]netip.Addr{
		synthesizeONNXRFC6052Address(localUse, rfc7050IPv4OnlyAddresses[0]),
	}); err == nil {
		t.Fatal("RFC 8215 local-use prefix must not be accepted as a public PREF64")
	}
}

func TestONNXPref64CacheCoalescesConcurrentLookupsAndExpires(t *testing.T) {
	prefix := netip.MustParsePrefix("2001:4860:1234:5678::/64")
	started := make(chan struct{})
	release := make(chan struct{})
	var startOnce sync.Once
	var lookups atomic.Int32
	resolver := onnxImageResolverFunc(func(_ context.Context, network, _ string) ([]netip.Addr, error) {
		if network != "ip6" {
			return nil, errors.New("unexpected lookup network")
		}
		lookups.Add(1)
		startOnce.Do(func() { close(started) })
		<-release
		return []netip.Addr{
			synthesizeONNXRFC6052Address(prefix, rfc7050IPv4OnlyAddresses[0]),
			synthesizeONNXRFC6052Address(prefix, rfc7050IPv4OnlyAddresses[1]),
		}, nil
	})
	cache := newPref64DiscoveryCache(resolver, time.Minute, time.Second, time.Second)
	now := time.Unix(1_700_000_000, 0)
	cache.now = func() time.Time { return now }

	assertConcurrentPref64CacheLookups(t, cache, prefix, &lookups, started, release)

	prefixes, err := cache.prefixesFor(context.Background())
	if err != nil || len(prefixes) != 1 {
		t.Fatalf("cached lookup failed: prefixes=%v err=%v", prefixes, err)
	}
	prefixes[0] = rfc6052WellKnownTranslationPrefix
	if again, _ := cache.prefixesFor(context.Background()); again[0] != prefix {
		t.Fatal("callers must not be able to mutate cached PREF64 state")
	}
	if got := lookups.Load(); got != 1 {
		t.Fatalf("unexpired cache performed %d lookups", got)
	}

	now = now.Add(time.Minute + time.Nanosecond)
	if _, err := cache.prefixesFor(context.Background()); err != nil {
		t.Fatalf("refresh after expiry: %v", err)
	}
	if got := lookups.Load(); got != 2 {
		t.Fatalf("expired cache must refresh once, got %d lookups", got)
	}
}

func assertConcurrentPref64CacheLookups(
	t *testing.T,
	cache *pref64DiscoveryCache,
	prefix netip.Prefix,
	lookups *atomic.Int32,
	started <-chan struct{},
	release chan<- struct{},
) {
	t.Helper()
	const callers = 16
	results := make(chan error, callers)
	var callersDone sync.WaitGroup
	callersDone.Add(callers)
	for i := 0; i < callers; i++ {
		go func() {
			defer callersDone.Done()
			prefixes, err := cache.prefixesFor(context.Background())
			if err == nil && (len(prefixes) != 1 || prefixes[0] != prefix) {
				err = errors.New("unexpected cached PREF64 result")
			}
			results <- err
		}()
	}
	<-started
	close(release)
	callersDone.Wait()
	close(results)
	for err := range results {
		if err != nil {
			t.Fatal(err)
		}
	}
	if got := lookups.Load(); got != 1 {
		t.Fatalf("concurrent cache miss must perform one lookup, got %d", got)
	}
}

func TestONNXImageDestinationsFailClosedWhenPref64CannotBeDiscovered(t *testing.T) {
	t.Parallel()

	var lookups atomic.Int32
	resolver := onnxImageResolverFunc(func(context.Context, string, string) ([]netip.Addr, error) {
		lookups.Add(1)
		return nil, errors.New("DNS unavailable")
	})
	cache := newPref64DiscoveryCache(resolver, time.Minute, time.Second, time.Second)
	if imageDestinationsArePublic(
		context.Background(),
		[]netip.Addr{netip.MustParseAddr("2606:4700:4700::1111")},
		cache,
	) {
		t.Fatal("public-looking IPv6 destination must fail closed without PREF64 discovery")
	}
	if lookups.Load() != 1 {
		t.Fatalf("IPv6 validation must attempt one discovery, got %d", lookups.Load())
	}
	if !imageDestinationsArePublic(
		context.Background(),
		[]netip.Addr{netip.MustParseAddr("1.1.1.1")},
		cache,
	) {
		t.Fatal("public IPv4 destination must not depend on PREF64 discovery")
	}
	if lookups.Load() != 1 {
		t.Fatal("IPv4-only validation must not perform PREF64 discovery")
	}
	if imageDestinationsArePublic(
		context.Background(),
		[]netip.Addr{netip.MustParseAddr("2606:4700:4700::1111")},
		cache,
	) {
		t.Fatal("cached discovery failure must continue to fail closed")
	}
	if lookups.Load() != 1 {
		t.Fatal("discovery failures must use the short failure cache")
	}
}

func TestONNXImageDestinationsUseDiscoveredPref64(t *testing.T) {
	t.Parallel()

	prefix := netip.MustParsePrefix("2001:4860:1234:5600::/56")
	var lookups atomic.Int32
	resolver := onnxImageResolverFunc(func(_ context.Context, network, _ string) ([]netip.Addr, error) {
		if network != "ip6" {
			return nil, errors.New("unexpected lookup network")
		}
		lookups.Add(1)
		return []netip.Addr{
			synthesizeONNXRFC6052Address(prefix, rfc7050IPv4OnlyAddresses[0]),
			synthesizeONNXRFC6052Address(prefix, rfc7050IPv4OnlyAddresses[1]),
		}, nil
	})
	cache := newPref64DiscoveryCache(resolver, time.Minute, time.Second, time.Second)
	private := synthesizeONNXRFC6052Address(prefix, netip.MustParseAddr("169.254.169.254"))
	if imageDestinationsArePublic(context.Background(), []netip.Addr{private}, cache) {
		t.Fatal("discovered PREF64 must expose and block an embedded metadata address")
	}
	public := synthesizeONNXRFC6052Address(prefix, netip.MustParseAddr("1.1.1.1"))
	if !imageDestinationsArePublic(context.Background(), []netip.Addr{public}, cache) {
		t.Fatal("discovered PREF64 must allow an embedded public address")
	}
	if lookups.Load() != 1 {
		t.Fatalf("successful discovery must be cached, got %d lookups", lookups.Load())
	}
}

func onnxRFC6052TestPrefixes() []netip.Prefix {
	return []netip.Prefix{
		netip.MustParsePrefix("2001:4860::/32"),
		netip.MustParsePrefix("2001:4860:1200::/40"),
		netip.MustParsePrefix("2001:4860:1234::/48"),
		netip.MustParsePrefix("2001:4860:1234:5600::/56"),
		netip.MustParsePrefix("2001:4860:1234:5678::/64"),
		netip.MustParsePrefix("2001:4860:1234:5678:ab:cdef::/96"),
	}
}

func synthesizeONNXRFC6052Address(prefix netip.Prefix, ipv4 netip.Addr) netip.Addr {
	bytes := prefix.Addr().As16()
	v4 := ipv4.As4()
	switch prefix.Bits() {
	case 32:
		copy(bytes[4:8], v4[:])
	case 40:
		copy(bytes[5:8], v4[:3])
		bytes[9] = v4[3]
	case 48:
		copy(bytes[6:8], v4[:2])
		copy(bytes[9:11], v4[2:])
	case 56:
		bytes[7] = v4[0]
		copy(bytes[9:12], v4[1:])
	case 64:
		copy(bytes[9:13], v4[:])
	case 96:
		copy(bytes[12:16], v4[:])
	default:
		panic("unsupported RFC 6052 prefix length")
	}
	bytes[8] = 0
	return netip.AddrFrom16(bytes)
}
