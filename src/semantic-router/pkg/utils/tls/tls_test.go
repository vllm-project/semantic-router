package tls

import (
	"crypto/tls"
	"crypto/x509"
	"testing"
	"time"
)

func TestCreateSelfSignedTLSCertificate(t *testing.T) {
	cert, err := CreateSelfSignedTLSCertificate()
	if err != nil {
		t.Fatalf("CreateSelfSignedTLSCertificate() returned error: %v", err)
	}

	// Verify certificate is not empty
	if len(cert.Certificate) == 0 {
		t.Fatal("Certificate chain is empty")
	}

	// Parse the certificate to verify structure
	x509Cert, err := x509.ParseCertificate(cert.Certificate[0])
	if err != nil {
		t.Fatalf("Failed to parse certificate: %v", err)
	}

	// Verify certificate properties
	t.Run("Organization", func(t *testing.T) {
		if len(x509Cert.Subject.Organization) == 0 {
			t.Error("Certificate has no organization")
		}
		if x509Cert.Subject.Organization[0] != "Inference Ext" {
			t.Errorf("Organization = %s, want 'Inference Ext'", x509Cert.Subject.Organization[0])
		}
	})

	t.Run("Validity Period", func(t *testing.T) {
		now := time.Now()
		if x509Cert.NotBefore.After(now) {
			t.Error("Certificate NotBefore is in the future")
		}
		if x509Cert.NotAfter.Before(now) {
			t.Error("Certificate NotAfter is in the past")
		}

		// Check that certificate is valid for approximately 10 years
		validityDuration := x509Cert.NotAfter.Sub(x509Cert.NotBefore)
		expectedDuration := time.Hour * 24 * 365 * 10 // 10 years
		tolerance := time.Hour * 24                   // 1 day tolerance

		if validityDuration < expectedDuration-tolerance || validityDuration > expectedDuration+tolerance {
			t.Errorf("Certificate validity duration = %v, want approximately %v", validityDuration, expectedDuration)
		}
	})

	t.Run("Key Usage", func(t *testing.T) {
		expectedKeyUsage := x509.KeyUsageKeyEncipherment | x509.KeyUsageDigitalSignature
		if x509Cert.KeyUsage != expectedKeyUsage {
			t.Errorf("KeyUsage = %v, want %v", x509Cert.KeyUsage, expectedKeyUsage)
		}
	})

	t.Run("Extended Key Usage", func(t *testing.T) {
		if len(x509Cert.ExtKeyUsage) == 0 {
			t.Fatal("No ExtKeyUsage found")
		}
		if x509Cert.ExtKeyUsage[0] != x509.ExtKeyUsageServerAuth {
			t.Errorf("ExtKeyUsage[0] = %v, want %v", x509Cert.ExtKeyUsage[0], x509.ExtKeyUsageServerAuth)
		}
	})

	t.Run("Serial Number", func(t *testing.T) {
		if x509Cert.SerialNumber == nil {
			t.Error("Serial number is nil")
		}
		if x509Cert.SerialNumber.Sign() <= 0 {
			t.Error("Serial number is not positive")
		}
	})

	t.Run("Basic Constraints", func(t *testing.T) {
		if !x509Cert.BasicConstraintsValid {
			t.Error("BasicConstraintsValid is false")
		}
	})
}

func TestCreateSelfSignedTLSCertificate_PrivateKey(t *testing.T) {
	cert, err := CreateSelfSignedTLSCertificate()
	if err != nil {
		t.Fatalf("CreateSelfSignedTLSCertificate() returned error: %v", err)
	}

	// Verify private key is present
	if cert.PrivateKey == nil {
		t.Fatal("Private key is nil")
	}

	// Try to use the certificate for TLS
	t.Run("TLS Config", func(t *testing.T) {
		config := &tls.Config{
			Certificates: []tls.Certificate{cert},
		}

		if len(config.Certificates) != 1 {
			t.Errorf("TLS config has %d certificates, want 1", len(config.Certificates))
		}
	})
}

func TestCreateSelfSignedTLSCertificate_Uniqueness(t *testing.T) {
	// Generate two certificates and verify they are different
	cert1, err1 := CreateSelfSignedTLSCertificate()
	if err1 != nil {
		t.Fatalf("First certificate generation failed: %v", err1)
	}

	cert2, err2 := CreateSelfSignedTLSCertificate()
	if err2 != nil {
		t.Fatalf("Second certificate generation failed: %v", err2)
	}

	// Parse both certificates
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

	// Verify certificates are different
	if string(cert1.Certificate[0]) == string(cert2.Certificate[0]) {
		t.Error("Two certificates are identical")
	}
}

func TestCreateSelfSignedTLSCertificate_SelfSigned(t *testing.T) {
	cert, err := CreateSelfSignedTLSCertificate()
	if err != nil {
		t.Fatalf("CreateSelfSignedTLSCertificate() returned error: %v", err)
	}

	x509Cert, err := x509.ParseCertificate(cert.Certificate[0])
	if err != nil {
		t.Fatalf("Failed to parse certificate: %v", err)
	}

	// Verify certificate is self-signed by checking if issuer equals subject
	if x509Cert.Issuer.String() != x509Cert.Subject.String() {
		t.Errorf("Certificate is not self-signed: Issuer=%s, Subject=%s",
			x509Cert.Issuer.String(), x509Cert.Subject.String())
	}

	// Verify certificate can be verified against itself
	roots := x509.NewCertPool()
	roots.AddCert(x509Cert)

	opts := x509.VerifyOptions{
		Roots:     roots,
		KeyUsages: []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth},
	}

	if _, err := x509Cert.Verify(opts); err != nil {
		t.Errorf("Certificate failed self-verification: %v", err)
	}
}

func TestCreateSelfSignedTLSCertificate_KeySize(t *testing.T) {
	cert, err := CreateSelfSignedTLSCertificate()
	if err != nil {
		t.Fatalf("CreateSelfSignedTLSCertificate() returned error: %v", err)
	}

	x509Cert, err := x509.ParseCertificate(cert.Certificate[0])
	if err != nil {
		t.Fatalf("Failed to parse certificate: %v", err)
	}

	// Verify RSA key size is 4096 bits
	if rsaPubKey, ok := x509Cert.PublicKey.(interface{ Size() int }); ok {
		keySize := rsaPubKey.Size() * 8 // Convert bytes to bits
		if keySize != 4096 {
			t.Errorf("RSA key size = %d bits, want 4096 bits", keySize)
		}
	}
}

func TestCreateSelfSignedTLSCertificate_CertificateChain(t *testing.T) {
	cert, err := CreateSelfSignedTLSCertificate()
	if err != nil {
		t.Fatalf("CreateSelfSignedTLSCertificate() returned error: %v", err)
	}

	// Verify certificate chain length
	if len(cert.Certificate) != 1 {
		t.Errorf("Certificate chain length = %d, want 1 (self-signed)", len(cert.Certificate))
	}
}

func TestCreateSelfSignedTLSCertificate_Timestamps(t *testing.T) {
	beforeCreation := time.Now().Add(-time.Minute) // 1 minute buffer
	cert, err := CreateSelfSignedTLSCertificate()
	afterCreation := time.Now().Add(time.Minute) // 1 minute buffer

	if err != nil {
		t.Fatalf("CreateSelfSignedTLSCertificate() returned error: %v", err)
	}

	x509Cert, err := x509.ParseCertificate(cert.Certificate[0])
	if err != nil {
		t.Fatalf("Failed to parse certificate: %v", err)
	}

	// Verify NotBefore is around creation time
	if x509Cert.NotBefore.Before(beforeCreation) || x509Cert.NotBefore.After(afterCreation) {
		t.Errorf("NotBefore timestamp is out of expected range: %v", x509Cert.NotBefore)
	}
}

func TestCreateSelfSignedTLSCertificate_MultipleCalls(t *testing.T) {
	// Test that multiple calls succeed
	numCerts := 5
	certs := make([]tls.Certificate, numCerts)
	serialNumbers := make(map[string]bool)

	for i := 0; i < numCerts; i++ {
		cert, err := CreateSelfSignedTLSCertificate()
		if err != nil {
			t.Fatalf("Call %d failed: %v", i, err)
		}
		certs[i] = cert

		x509Cert, err := x509.ParseCertificate(cert.Certificate[0])
		if err != nil {
			t.Fatalf("Failed to parse certificate %d: %v", i, err)
		}

		serialNum := x509Cert.SerialNumber.String()
		if serialNumbers[serialNum] {
			t.Errorf("Duplicate serial number found: %s", serialNum)
		}
		serialNumbers[serialNum] = true
	}

	if len(serialNumbers) != numCerts {
		t.Errorf("Expected %d unique serial numbers, got %d", numCerts, len(serialNumbers))
	}
}

func BenchmarkCreateSelfSignedTLSCertificate(b *testing.B) {
	for i := 0; i < b.N; i++ {
		_, err := CreateSelfSignedTLSCertificate()
		if err != nil {
			b.Fatalf("Certificate generation failed: %v", err)
		}
	}
}
