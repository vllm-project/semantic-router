package backendegress

import (
	"context"
	"fmt"
	"net"
	"net/netip"
	"sort"
)

type Resolver interface {
	LookupNetIP(context.Context, string, string) ([]netip.Addr, error)
}

type Guard struct {
	Policy   Policy
	Resolver Resolver
}

type ResolvedTarget struct {
	Target
	Addresses []netip.Addr
}

func (g Guard) Resolve(ctx context.Context, origin string) (ResolvedTarget, error) {
	target, err := g.Policy.AuthorizeOrigin(origin)
	if err != nil {
		return ResolvedTarget{}, err
	}
	addresses, err := g.resolveHost(ctx, target.Host)
	if err != nil {
		return ResolvedTarget{}, err
	}
	for _, address := range addresses {
		if err := target.rule.authorizeAddress(address); err != nil {
			return ResolvedTarget{}, fmt.Errorf("backend origin resolved to a denied address: %w", err)
		}
	}
	return ResolvedTarget{Target: target, Addresses: addresses}, nil
}

func (g Guard) resolveHost(ctx context.Context, host string) ([]netip.Addr, error) {
	if literal, err := netip.ParseAddr(host); err == nil {
		return []netip.Addr{literal.Unmap()}, nil
	}
	resolver := g.Resolver
	if resolver == nil {
		resolver = net.DefaultResolver
	}
	values, err := resolver.LookupNetIP(ctx, "ip", host)
	if err != nil {
		return nil, fmt.Errorf("resolve backend origin: %w", err)
	}
	seen := make(map[netip.Addr]struct{}, len(values))
	addresses := make([]netip.Addr, 0, len(values))
	for _, value := range values {
		if !value.IsValid() {
			return nil, fmt.Errorf("backend origin returned an invalid address")
		}
		address := value.Unmap()
		if _, duplicate := seen[address]; duplicate {
			continue
		}
		seen[address] = struct{}{}
		addresses = append(addresses, address)
	}
	if len(addresses) == 0 || len(addresses) > maximumResolvedIPs {
		return nil, fmt.Errorf("backend origin resolved address count is invalid")
	}
	sort.Slice(addresses, func(left, right int) bool { return addresses[left].Less(addresses[right]) })
	return addresses, nil
}

func (r hostRule) authorizeAddress(address netip.Addr) error {
	address = address.Unmap()
	if !address.IsValid() || criticalAddress(address) {
		return fmt.Errorf("critical local or metadata range")
	}
	if address.IsPrivate() {
		for _, allowed := range r.privateCIDRs {
			if allowed.Contains(address) {
				return nil
			}
		}
		return fmt.Errorf("private range has no exact policy exception")
	}
	if !address.IsGlobalUnicast() {
		return fmt.Errorf("address is not global unicast")
	}
	return nil
}

func criticalAddress(address netip.Addr) bool {
	address = address.Unmap()
	if !address.IsValid() || address.IsUnspecified() || address.IsLoopback() ||
		address.IsMulticast() || address.IsLinkLocalUnicast() || address.IsLinkLocalMulticast() {
		return true
	}
	for _, blocked := range criticalPrefixes {
		if blocked.Contains(address) {
			return true
		}
	}
	return false
}

var criticalPrefixes = []netip.Prefix{
	// Construct the cloud metadata address as a typed value so it can never be
	// confused with an outbound URL or credential reference.
	netip.PrefixFrom(netip.AddrFrom4([4]byte{169, 254, 169, 254}), 32),
	netip.MustParsePrefix("100.100.100.200/32"),
	netip.MustParsePrefix("fd00:ec2::254/128"),
}
