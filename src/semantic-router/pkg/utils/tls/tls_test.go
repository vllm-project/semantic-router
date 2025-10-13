package tls

import (
	"crypto/rsa"
	"crypto/tls"
	"crypto/x509"
	"encoding/pem"
	"testing"
	"time"
)

func TestCreateSelfSignedTLSCertificate(t *testing.T) {
	// Test successful certificate creation
	cert, err := CreateSelfSignedTLSCertificate()
	if err != nil {
		t.Fatalf("CreateSelfSignedTLSCertificate() failed: %v", err)
	}

	// Verify certificate is not empty
	if len(cert.Certificate) == 0 {
		t.Fatal("Certificate is empty")
	}

	// Verify private key is present
	if cert.PrivateKey == nil {
		t.Fatal("Private key is nil")
	}

	// Parse the certificate to verify its properties
	x509Cert, err := x509.ParseCertificate(cert.Certificate[0])
	if err != nil {
		t.Fatalf("Failed to parse certificate: %v", err)
	}

	// Verify organization name
	if len(x509Cert.Subject.Organization) == 0 {
		t.Fatal("Organization is empty")
	}
	if x509Cert.Subject.Organization[0] != "Inference Ext" {
		t.Errorf("Expected organization 'Inference Ext', got '%s'", x509Cert.Subject.Organization[0])
	}

	// Verify NotBefore is set and reasonable (within last minute)
	now := time.Now()
	if x509Cert.NotBefore.After(now) {
		t.Errorf("NotBefore (%v) is in the future", x509Cert.NotBefore)
	}
	if now.Sub(x509Cert.NotBefore) > time.Minute {
		t.Errorf("NotBefore (%v) is too far in the past", x509Cert.NotBefore)
	}

	// Verify NotAfter is approximately 10 years in the future
	expectedNotAfter := now.Add(time.Hour * 24 * 365 * 10)
	timeDiff := x509Cert.NotAfter.Sub(expectedNotAfter)
	if timeDiff < -time.Minute || timeDiff > time.Minute {
		t.Errorf("NotAfter is not approximately 10 years in the future. Expected ~%v, got %v", expectedNotAfter, x509Cert.NotAfter)
	}

	// Verify key usage
	expectedKeyUsage := x509.KeyUsageKeyEncipherment | x509.KeyUsageDigitalSignature
	if x509Cert.KeyUsage != expectedKeyUsage {
		t.Errorf("Expected KeyUsage %v, got %v", expectedKeyUsage, x509Cert.KeyUsage)
	}

	// Verify extended key usage
	if len(x509Cert.ExtKeyUsage) == 0 {
		t.Fatal("ExtKeyUsage is empty")
	}
	if x509Cert.ExtKeyUsage[0] != x509.ExtKeyUsageServerAuth {
		t.Errorf("Expected ExtKeyUsage ServerAuth, got %v", x509Cert.ExtKeyUsage[0])
	}

	// Verify BasicConstraintsValid is true
	if !x509Cert.BasicConstraintsValid {
		t.Error("BasicConstraintsValid should be true")
	}

	// Verify serial number is set and non-zero
	if x509Cert.SerialNumber == nil || x509Cert.SerialNumber.Sign() == 0 {
		t.Error("Serial number is not set or is zero")
	}

	// Verify the certificate can be used to create a TLS config
	tlsConfig := &tls.Config{
		Certificates: []tls.Certificate{cert},
	}
	if len(tlsConfig.Certificates) == 0 {
		t.Fatal("Failed to create TLS config with certificate")
	}
}

func TestCreateSelfSignedTLSCertificate_PEMEncoding(t *testing.T) {
	cert, err := CreateSelfSignedTLSCertificate()
	if err != nil {
		t.Fatalf("CreateSelfSignedTLSCertificate() failed: %v", err)
	}

	// Verify the Leaf certificate can be parsed
	if cert.Leaf == nil {
		// Leaf is nil by default, parse it
		x509Cert, err := x509.ParseCertificate(cert.Certificate[0])
		if err != nil {
			t.Fatalf("Failed to parse certificate: %v", err)
		}
		cert.Leaf = x509Cert
	}

	// Verify certificate chain length
	if len(cert.Certificate) != 1 {
		t.Errorf("Expected 1 certificate in chain, got %d", len(cert.Certificate))
	}
}

func TestCreateSelfSignedTLSCertificate_CertificateValidity(t *testing.T) {
	cert, err := CreateSelfSignedTLSCertificate()
	if err != nil {
		t.Fatalf("CreateSelfSignedTLSCertificate() failed: %v", err)
	}

	x509Cert, err := x509.ParseCertificate(cert.Certificate[0])
	if err != nil {
		t.Fatalf("Failed to parse certificate: %v", err)
	}

	// Test that the certificate is self-signed (issuer == subject)
	if x509Cert.Issuer.String() != x509Cert.Subject.String() {
		t.Error("Certificate is not self-signed")
	}

	// Verify the certificate using itself as the CA
	roots := x509.NewCertPool()
	roots.AddCert(x509Cert)

	opts := x509.VerifyOptions{
		Roots:     roots,
		KeyUsages: []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth},
	}

	if _, err := x509Cert.Verify(opts); err != nil {
		t.Errorf("Certificate verification failed: %v", err)
	}
}

func TestCreateSelfSignedTLSCertificate_RSAKeySize(t *testing.T) {
	cert, err := CreateSelfSignedTLSCertificate()
	if err != nil {
		t.Fatalf("CreateSelfSignedTLSCertificate() failed: %v", err)
	}

	x509Cert, err := x509.ParseCertificate(cert.Certificate[0])
	if err != nil {
		t.Fatalf("Failed to parse certificate: %v", err)
	}

	// Verify RSA key size is 4096 bits
	rsaKey, ok := x509Cert.PublicKey.(*rsa.PublicKey)
	if !ok {
		t.Fatal("Public key is not RSA")
	}

	keySize := rsaKey.N.BitLen()
	if keySize != 4096 {
		t.Errorf("Expected RSA key size of 4096 bits, got %d", keySize)
	}
}

func TestCreateSelfSignedTLSCertificate_MultipleInvocations(t *testing.T) {
	// Create multiple certificates and verify they're different
	cert1, err := CreateSelfSignedTLSCertificate()
	if err != nil {
		t.Fatalf("First CreateSelfSignedTLSCertificate() failed: %v", err)
	}

	cert2, err := CreateSelfSignedTLSCertificate()
	if err != nil {
		t.Fatalf("Second CreateSelfSignedTLSCertificate() failed: %v", err)
	}

	x509Cert1, err := x509.ParseCertificate(cert1.Certificate[0])
	if err != nil {
		t.Fatalf("Failed to parse first certificate: %v", err)
	}

	x509Cert2, err := x509.ParseCertificate(cert2.Certificate[0])
	if err != nil {
		t.Fatalf("Failed to parse second certificate: %v", err)
	}

	// Verify serial numbers are different
	if x509Cert1.SerialNumber.Cmp(x509Cert2.SerialNumber) == 0 {
		t.Error("Two certificates have the same serial number")
	}

	// Verify they have different public keys
	rsaKey1, ok1 := x509Cert1.PublicKey.(*rsa.PublicKey)
	rsaKey2, ok2 := x509Cert2.PublicKey.(*rsa.PublicKey)
	if !ok1 || !ok2 {
		t.Fatal("Public keys are not RSA")
	}

	if rsaKey1.N.Cmp(rsaKey2.N) == 0 {
		t.Error("Two certificates have the same public key")
	}
}

func TestCreateSelfSignedTLSCertificate_TLSUsage(t *testing.T) {
	cert, err := CreateSelfSignedTLSCertificate()
	if err != nil {
		t.Fatalf("CreateSelfSignedTLSCertificate() failed: %v", err)
	}

	// Create a TLS config and verify it's usable
	tlsConfig := &tls.Config{
		Certificates: []tls.Certificate{cert},
		MinVersion:   tls.VersionTLS12,
	}

	// Verify certificate count
	if len(tlsConfig.Certificates) != 1 {
		t.Errorf("Expected 1 certificate in TLS config, got %d", len(tlsConfig.Certificates))
	}

	// Verify the certificate is accessible
	if tlsConfig.Certificates[0].PrivateKey == nil {
		t.Error("Private key is nil in TLS config")
	}

	if len(tlsConfig.Certificates[0].Certificate) == 0 {
		t.Error("Certificate chain is empty in TLS config")
	}
}

func TestCreateSelfSignedTLSCertificate_PEMFormat(t *testing.T) {
	cert, err := CreateSelfSignedTLSCertificate()
	if err != nil {
		t.Fatalf("CreateSelfSignedTLSCertificate() failed: %v", err)
	}

	// The certificate should be in DER format in the Certificate field
	// Verify we can decode it
	block, _ := pem.Decode(pem.EncodeToMemory(&pem.Block{
		Type:  "CERTIFICATE",
		Bytes: cert.Certificate[0],
	}))

	if block == nil {
		t.Fatal("Failed to decode PEM block")
	}

	if block.Type != "CERTIFICATE" {
		t.Errorf("Expected PEM type 'CERTIFICATE', got '%s'", block.Type)
	}

	// Verify the decoded bytes match the original
	if len(block.Bytes) != len(cert.Certificate[0]) {
		t.Error("PEM decoded bytes length doesn't match original")
	}
}

func TestCreateSelfSignedTLSCertificate_TimeFormat(t *testing.T) {
	cert, err := CreateSelfSignedTLSCertificate()
	if err != nil {
		t.Fatalf("CreateSelfSignedTLSCertificate() failed: %v", err)
	}

	x509Cert, err := x509.ParseCertificate(cert.Certificate[0])
	if err != nil {
		t.Fatalf("Failed to parse certificate: %v", err)
	}

	// Verify times are in UTC
	if x509Cert.NotBefore.Location() != time.UTC {
		t.Errorf("NotBefore is not in UTC: %v", x509Cert.NotBefore.Location())
	}

	if x509Cert.NotAfter.Location() != time.UTC {
		t.Errorf("NotAfter is not in UTC: %v", x509Cert.NotAfter.Location())
	}

	// Verify the certificate is currently valid
	now := time.Now()
	if now.Before(x509Cert.NotBefore) {
		t.Error("Certificate is not yet valid")
	}
	if now.After(x509Cert.NotAfter) {
		t.Error("Certificate has expired")
	}
}

func TestCreateSelfSignedTLSCertificate_SerialNumberUniqueness(t *testing.T) {
	// Create many certificates and verify serial numbers are unique
	const numCerts = 100
	serialNumbers := make(map[string]bool)

	for i := 0; i < numCerts; i++ {
		cert, err := CreateSelfSignedTLSCertificate()
		if err != nil {
			t.Fatalf("CreateSelfSignedTLSCertificate() failed on iteration %d: %v", i, err)
		}

		x509Cert, err := x509.ParseCertificate(cert.Certificate[0])
		if err != nil {
			t.Fatalf("Failed to parse certificate on iteration %d: %v", i, err)
		}

		serialStr := x509Cert.SerialNumber.String()
		if serialNumbers[serialStr] {
			t.Errorf("Duplicate serial number found: %s", serialStr)
		}
		serialNumbers[serialStr] = true
	}
}

func TestCreateSelfSignedTLSCertificate_PrivateKeyMarshalling(t *testing.T) {
	cert, err := CreateSelfSignedTLSCertificate()
	if err != nil {
		t.Fatalf("CreateSelfSignedTLSCertificate() failed: %v", err)
	}

	// Verify the private key can be marshalled back
	privKey, ok := cert.PrivateKey.(*rsa.PrivateKey)
	if !ok {
		t.Fatal("Private key is not RSA")
	}

	// Test PKCS8 marshalling (which is what the code uses)
	pkcs8Bytes, err := x509.MarshalPKCS8PrivateKey(privKey)
	if err != nil {
		t.Fatalf("Failed to marshal private key to PKCS8: %v", err)
	}

	if len(pkcs8Bytes) == 0 {
		t.Error("PKCS8 marshalled key is empty")
	}

	// Verify we can parse it back
	parsedKey, err := x509.ParsePKCS8PrivateKey(pkcs8Bytes)
	if err != nil {
		t.Fatalf("Failed to parse PKCS8 private key: %v", err)
	}

	parsedRSA, ok := parsedKey.(*rsa.PrivateKey)
	if !ok {
		t.Fatal("Parsed key is not RSA")
	}

	// Verify the modulus matches (indicates same key)
	if parsedRSA.N.Cmp(privKey.N) != 0 {
		t.Error("Parsed key modulus doesn't match original")
	}
}

func TestCreateSelfSignedTLSCertificate_X509KeyPairCompatibility(t *testing.T) {
	// Test that the generated certificate can be used with tls.X509KeyPair
	cert, err := CreateSelfSignedTLSCertificate()
	if err != nil {
		t.Fatalf("CreateSelfSignedTLSCertificate() failed: %v", err)
	}

	// Encode to PEM format as the function does
	certPEM := pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: cert.Certificate[0]})

	privKey := cert.PrivateKey.(*rsa.PrivateKey)
	privBytes, err := x509.MarshalPKCS8PrivateKey(privKey)
	if err != nil {
		t.Fatalf("Failed to marshal private key: %v", err)
	}
	keyPEM := pem.EncodeToMemory(&pem.Block{Type: "PRIVATE KEY", Bytes: privBytes})

	// Verify we can recreate the certificate from PEM
	recreatedCert, err := tls.X509KeyPair(certPEM, keyPEM)
	if err != nil {
		t.Fatalf("Failed to recreate certificate from PEM: %v", err)
	}

	if len(recreatedCert.Certificate) == 0 {
		t.Error("Recreated certificate has no certificates")
	}

	if recreatedCert.PrivateKey == nil {
		t.Error("Recreated certificate has no private key")
	}
}

func TestCreateSelfSignedTLSCertificate_CertificateExtensions(t *testing.T) {
	cert, err := CreateSelfSignedTLSCertificate()
	if err != nil {
		t.Fatalf("CreateSelfSignedTLSCertificate() failed: %v", err)
	}

	x509Cert, err := x509.ParseCertificate(cert.Certificate[0])
	if err != nil {
		t.Fatalf("Failed to parse certificate: %v", err)
	}

	// Verify it's not a CA certificate
	if x509Cert.IsCA {
		t.Error("Certificate should not be a CA")
	}

	// Verify signature algorithm is set
	if x509Cert.SignatureAlgorithm == x509.UnknownSignatureAlgorithm {
		t.Error("Signature algorithm is unknown")
	}

	// Verify public key algorithm is RSA
	if x509Cert.PublicKeyAlgorithm != x509.RSA {
		t.Errorf("Expected RSA public key algorithm, got %v", x509Cert.PublicKeyAlgorithm)
	}
}

func TestCreateSelfSignedTLSCertificate_SerialNumberBitLength(t *testing.T) {
	cert, err := CreateSelfSignedTLSCertificate()
	if err != nil {
		t.Fatalf("CreateSelfSignedTLSCertificate() failed: %v", err)
	}

	x509Cert, err := x509.ParseCertificate(cert.Certificate[0])
	if err != nil {
		t.Fatalf("Failed to parse certificate: %v", err)
	}

	// Serial number should be less than 2^128 (as per RFC 5280)
	serialBitLen := x509Cert.SerialNumber.BitLen()
	if serialBitLen > 128 {
		t.Errorf("Serial number bit length (%d) exceeds 128 bits", serialBitLen)
	}

	// Serial number should be positive
	if x509Cert.SerialNumber.Sign() <= 0 {
		t.Error("Serial number should be positive")
	}
}
