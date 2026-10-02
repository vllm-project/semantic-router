//go:build !windows && cgo && (amd64 || arm64 || riscv64)

package candle_binding

import (
	"errors"
	"runtime"
	"testing"
)

func TestNativeClassifierFFIIsLinked(t *testing.T) {
	if runtime.GOARCH != "riscv64" {
		t.Skip("riscv64 Candle FFI smoke runs under qemu-user, not the host architecture")
	}

	cases := []struct {
		name string
		call func() error
	}{
		{
			name: "InitClassifier",
			call: func() error {
				return InitClassifier("/no-such-riscv-classifier", 2, true)
			},
		},
		{
			name: "InitJailbreakClassifier",
			call: func() error {
				return InitJailbreakClassifier("/no-such-riscv-jailbreak", 2, true)
			},
		},
		{
			name: "InitPIIClassifier",
			call: func() error {
				return InitPIIClassifier("/no-such-riscv-pii", 2, true)
			},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			err := tc.call()
			if err == nil {
				t.Fatalf("%s: expected native missing-model error, got nil", tc.name)
			}
			if errors.Is(err, ErrBackendUnavailable) {
				t.Fatalf("%s: compiled the unavailable stub instead of the Candle FFI: %v", tc.name, err)
			}
		})
	}
}
