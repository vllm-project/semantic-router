package connectortest

import (
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/tls"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/pem"
	"math/big"
	"os"
	"path/filepath"
	"testing"
	"time"
)

// PrivateCA is a private PKI as a company runs one: a root signs an issuing
// intermediate, and the intermediate signs the server certificate. Its server
// sends only its own certificate, so a client verifies it only when its
// trust bundle holds the intermediate.
type PrivateCA struct {
	// Server is the leaf certificate and key for ServerName, without the
	// intermediate.
	Server tls.Certificate
	// ServerName is the DNS name the server certificate is valid for.
	ServerName string
	// ChainFile holds the intermediate and the root, RootFile the root alone.
	ChainFile string
	RootFile  string
	// Client is a client certificate the intermediate issued, in ClientCert
	// and ClientKey, and ClientPool is the pool that verifies it.
	ClientCert string
	ClientKey  string
	ClientPool *x509.CertPool
}

// NewPrivateCA issues the PKI and writes its PEM files under a test
// directory.
func NewPrivateCA(t testing.TB, serverName string) *PrivateCA {
	t.Helper()
	root, rootKey := issue(t, &x509.Certificate{
		Subject:               pkix.Name{CommonName: "Example Corp Root CA"},
		IsCA:                  true,
		BasicConstraintsValid: true,
		KeyUsage:              x509.KeyUsageCertSign | x509.KeyUsageCRLSign,
	}, nil, nil)
	intermediate, intermediateKey := issue(t, &x509.Certificate{
		Subject:               pkix.Name{CommonName: "Example Corp Issuing CA"},
		IsCA:                  true,
		BasicConstraintsValid: true,
		MaxPathLenZero:        true,
		KeyUsage:              x509.KeyUsageCertSign | x509.KeyUsageCRLSign,
	}, root, rootKey)
	leaf, leafKey := issue(t, &x509.Certificate{
		Subject:     pkix.Name{CommonName: serverName},
		DNSNames:    []string{serverName},
		KeyUsage:    x509.KeyUsageDigitalSignature,
		ExtKeyUsage: []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth},
	}, intermediate, intermediateKey)
	client, clientKey := issue(t, &x509.Certificate{
		Subject:     pkix.Name{CommonName: "router.internal.example"},
		KeyUsage:    x509.KeyUsageDigitalSignature,
		ExtKeyUsage: []x509.ExtKeyUsage{x509.ExtKeyUsageClientAuth},
	}, intermediate, intermediateKey)

	dir := t.TempDir()
	ca := &PrivateCA{
		Server:     tls.Certificate{Certificate: [][]byte{leaf.Raw}, PrivateKey: leafKey, Leaf: leaf},
		ServerName: serverName,
		ChainFile:  filepath.Join(dir, "example-corp-chain.pem"),
		RootFile:   filepath.Join(dir, "example-corp-root.pem"),
		ClientCert: filepath.Join(dir, "router.crt"),
		ClientKey:  filepath.Join(dir, "router.key"),
		ClientPool: x509.NewCertPool(),
	}
	ca.ClientPool.AddCert(intermediate)
	writePEM(t, ca.ChainFile, certificatePEM(intermediate), certificatePEM(root))
	writePEM(t, ca.RootFile, certificatePEM(root))
	writePEM(t, ca.ClientCert, certificatePEM(client))
	writePEM(t, ca.ClientKey, keyPEM(t, clientKey))
	return ca
}

func issue(t testing.TB, template *x509.Certificate, parent *x509.Certificate, parentKey *ecdsa.PrivateKey) (*x509.Certificate, *ecdsa.PrivateKey) {
	t.Helper()
	key, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		t.Fatal(err)
	}
	serial, err := rand.Int(rand.Reader, new(big.Int).Lsh(big.NewInt(1), 62))
	if err != nil {
		t.Fatal(err)
	}
	template.SerialNumber = serial
	template.NotBefore = time.Now().Add(-time.Hour)
	template.NotAfter = time.Now().Add(time.Hour)
	if parent == nil {
		parent, parentKey = template, key
	}
	der, err := x509.CreateCertificate(rand.Reader, template, parent, &key.PublicKey, parentKey)
	if err != nil {
		t.Fatal(err)
	}
	certificate, err := x509.ParseCertificate(der)
	if err != nil {
		t.Fatal(err)
	}
	return certificate, key
}

func certificatePEM(certificate *x509.Certificate) []byte {
	return pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: certificate.Raw})
}

func keyPEM(t testing.TB, key *ecdsa.PrivateKey) []byte {
	t.Helper()
	der, err := x509.MarshalPKCS8PrivateKey(key)
	if err != nil {
		t.Fatal(err)
	}
	return pem.EncodeToMemory(&pem.Block{Type: "PRIVATE KEY", Bytes: der})
}

func writePEM(t testing.TB, path string, blocks ...[]byte) {
	t.Helper()
	var data []byte
	for _, block := range blocks {
		data = append(data, block...)
	}
	if err := os.WriteFile(path, data, 0o600); err != nil {
		t.Fatal(err)
	}
}
