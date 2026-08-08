package onnx_binding

import "testing"

// TestABILayoutAgreement is the model-free cross-language ABI receipt: for
// every FFI struct shared between the cgo preamble and the Rust #[repr(C)]
// definitions, the size, alignment, and every field offset must agree
// byte-for-byte. A field reorder or type change on either side compiles
// cleanly but fails here, before it can corrupt memory at runtime.
func TestABILayoutAgreement(t *testing.T) {
	pairs := abiLayoutPairs()
	if len(pairs) == 0 {
		t.Fatal("No ABI layout pairs registered")
	}

	for _, pair := range pairs {
		t.Run(pair.name, func(t *testing.T) {
			expected := 2 + len(pair.fields) // size, align, one offset per field
			if len(pair.rust) != expected || len(pair.cgo) != expected {
				t.Fatalf("Receipt length mismatch: rust=%d cgo=%d expected=%d (field list stale?)",
					len(pair.rust), len(pair.cgo), expected)
			}

			labels := append([]string{"size", "align"}, pair.fields...)
			for i, label := range labels {
				if pair.rust[i] != pair.cgo[i] {
					t.Errorf("%s.%s: rust=%d cgo=%d — #[repr(C)] and cgo preamble disagree",
						pair.name, label, pair.rust[i], pair.cgo[i])
				}
			}
		})
	}
}
