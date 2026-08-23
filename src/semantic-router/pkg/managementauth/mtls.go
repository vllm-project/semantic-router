package managementauth

import (
	"bytes"
	"context"
	"crypto/sha256"
	"crypto/tls"
	"encoding/hex"
	"net/url"
	"strings"
	"time"
)

type VerifiedMTLSEvidence struct {
	MatcherKind         string
	MatcherValue        string
	CertificateNotAfter time.Time
}

type VerifiedMTLSIdentity struct {
	PrincipalID       string
	MappingID         string
	WorkloadClass     string
	SourceAssuredAt   time.Time
	EvidenceExpiresAt time.Time
}

type MTLSIdentityResolver interface {
	ResolveMTLSIdentity(context.Context, VerifiedMTLSEvidence, time.Time) (VerifiedMTLSIdentity, error)
}

// VerifiedMTLSEvidenceFromConnection derives exactly one selector from the
// listener-verified leaf certificate. It never accepts forwarded headers or an
// unverified PeerCertificates-only connection state.
func VerifiedMTLSEvidenceFromConnection(state *tls.ConnectionState) (VerifiedMTLSEvidence, error) {
	if state == nil || len(state.VerifiedChains) == 0 || len(state.PeerCertificates) == 0 ||
		len(state.VerifiedChains[0]) == 0 ||
		!bytes.Equal(state.VerifiedChains[0][0].Raw, state.PeerCertificates[0].Raw) {
		return VerifiedMTLSEvidence{}, ErrAuthenticationDenied
	}
	leaf := state.PeerCertificates[0]
	spiffe := make([]string, 0, len(leaf.URIs))
	otherURIs := make([]string, 0, len(leaf.URIs))
	for _, candidate := range leaf.URIs {
		if candidate == nil {
			continue
		}
		canonical := candidate.String()
		if candidate.Scheme == "spiffe" {
			spiffe = append(spiffe, canonical)
		} else {
			otherURIs = append(otherURIs, canonical)
		}
	}
	if len(spiffe) > 0 {
		return uniqueMTLSSelector("spiffe_id", spiffe, leaf.NotAfter)
	}
	if len(otherURIs) > 0 {
		return uniqueMTLSSelector("san_uri", otherURIs, leaf.NotAfter)
	}
	if len(leaf.DNSNames) > 0 {
		names := make([]string, len(leaf.DNSNames))
		for index, name := range leaf.DNSNames {
			names[index] = strings.ToLower(strings.TrimSuffix(name, "."))
		}
		return uniqueMTLSSelector("san_dns", names, leaf.NotAfter)
	}
	if len(leaf.RawSubject) == 0 {
		return VerifiedMTLSEvidence{}, ErrAuthenticationDenied
	}
	digest := sha256.Sum256(leaf.RawSubject)
	return VerifiedMTLSEvidence{
		MatcherKind: "subject_dn_sha256", MatcherValue: hex.EncodeToString(digest[:]),
		CertificateNotAfter: leaf.NotAfter.UTC(),
	}, nil
}

func uniqueMTLSSelector(kind string, values []string, notAfter time.Time) (VerifiedMTLSEvidence, error) {
	if len(values) != 1 || values[0] == "" {
		return VerifiedMTLSEvidence{}, ErrAuthenticationDenied
	}
	if kind == "spiffe_id" || kind == "san_uri" {
		parsed, err := url.Parse(values[0])
		if err != nil || !parsed.IsAbs() || parsed.String() != values[0] {
			return VerifiedMTLSEvidence{}, ErrAuthenticationDenied
		}
	}
	if notAfter.IsZero() {
		return VerifiedMTLSEvidence{}, ErrAuthenticationDenied
	}
	return VerifiedMTLSEvidence{
		MatcherKind: kind, MatcherValue: values[0], CertificateNotAfter: notAfter.UTC(),
	}, nil
}
