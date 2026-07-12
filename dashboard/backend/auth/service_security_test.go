package auth

import (
	"bytes"
	"encoding/base64"
	"errors"
	"strings"
	"testing"
	"testing/iotest"
	"time"
)

const testJWTSecret = "0123456789abcdef0123456789abcdef"

func TestNewServiceFailsClosedWhenJWTEntropyIsUnavailable(t *testing.T) {
	t.Parallel()

	entropyErr := errors.New("entropy unavailable")
	svc, err := newServiceWithEntropy(nil, "", 1, iotest.ErrReader(entropyErr))
	if svc != nil {
		t.Fatal("newServiceWithEntropy() returned a service after entropy failure")
	}
	if !errors.Is(err, entropyErr) {
		t.Fatalf("newServiceWithEntropy() error = %v, want entropy failure", err)
	}
	if err == nil || !strings.Contains(err.Error(), "JWT signing secret") {
		t.Fatalf("newServiceWithEntropy() error = %v, want safe construction context", err)
	}
}

func TestNewServiceRejectsUnsafeConfiguredJWTSecrets(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name   string
		secret string
	}{
		{name: "31 bytes", secret: strings.Repeat("x", minimumJWTSecretBytes-1)},
		{name: "single repeated character", secret: strings.Repeat("x", minimumJWTSecretBytes)},
		{name: "repeated placeholder", secret: strings.Repeat("change-me-", 4)},
		{name: "known placeholder", secret: "replace-with-random-secret-before-production"},
		{name: "whitespace only", secret: strings.Repeat(" ", minimumJWTSecretBytes)},
		{name: "leading whitespace", secret: " " + testJWTSecret},
		{name: "trailing whitespace", secret: testJWTSecret + "\n"},
		{name: "interior control", secret: testJWTSecret[:16] + "\x00" + testJWTSecret[16:]},
		{name: "invalid UTF-8", secret: testJWTSecret + string([]byte{0xff})},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			svc, err := newServiceWithEntropy(
				nil,
				test.secret,
				1,
				iotest.ErrReader(errors.New("configured secrets must not read entropy")),
			)
			if svc != nil {
				t.Fatal("newServiceWithEntropy() returned a service for an unsafe secret")
			}
			if err == nil {
				t.Fatal("newServiceWithEntropy() accepted an unsafe secret")
			}
			if !strings.Contains(err.Error(), jwtSecretGenerationAdvice) {
				t.Fatalf("error = %q, want CSPRNG generation guidance", err)
			}
			if strings.Contains(err.Error(), test.secret) {
				t.Fatalf("error exposed configured secret: %q", err)
			}
		})
	}
}

func TestNewServiceAcceptsMinimumConfiguredJWTSecretWithoutReadingEntropy(t *testing.T) {
	t.Parallel()

	svc, err := newServiceWithEntropy(
		nil,
		testJWTSecret,
		1,
		iotest.ErrReader(errors.New("must not be read")),
	)
	if err != nil {
		t.Fatalf("newServiceWithEntropy() error = %v", err)
	}
	if got := string(svc.jwtSecret); got != testJWTSecret {
		t.Fatalf("JWT secret = %q, want configured value", got)
	}
}

func TestNewServiceGeneratesThirtyTwoRandomBytesWhenSecretIsEmpty(t *testing.T) {
	t.Parallel()

	wantEntropy := bytes.Repeat([]byte{0xa5}, minimumJWTSecretBytes)
	svc, err := newServiceWithEntropy(nil, "", 1, bytes.NewReader(wantEntropy))
	if err != nil {
		t.Fatalf("newServiceWithEntropy() error = %v", err)
	}
	decoded, err := base64.RawStdEncoding.DecodeString(string(svc.jwtSecret))
	if err != nil {
		t.Fatalf("decode generated JWT secret: %v", err)
	}
	if !bytes.Equal(decoded, wantEntropy) {
		t.Fatalf("decoded JWT secret has %d bytes, want the 32-byte entropy input", len(decoded))
	}
}

func TestNewServiceBoundsJWTExpiryBeforeDurationConversion(t *testing.T) {
	t.Parallel()

	for _, ttlHours := range []int{-1, maximumJWTExpiryHours + 1, int(^uint(0) >> 1)} {
		svc, err := newServiceWithEntropy(nil, testJWTSecret, ttlHours, nil)
		if svc != nil || err == nil {
			t.Fatalf("newServiceWithEntropy(ttl=%d) = (%#v, %v), want rejection", ttlHours, svc, err)
		}
	}
	svc, err := newServiceWithEntropy(nil, testJWTSecret, maximumJWTExpiryHours, nil)
	if err != nil {
		t.Fatalf("newServiceWithEntropy(max ttl) error = %v", err)
	}
	if svc.ttlDuration != time.Duration(maximumJWTExpiryHours)*time.Hour {
		t.Fatalf("ttlDuration = %v, want %dh", svc.ttlDuration, maximumJWTExpiryHours)
	}
}
