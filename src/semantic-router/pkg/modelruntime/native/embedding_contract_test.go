package native

import "testing"

func TestNativeOnlyDimensionContract(t *testing.T) {
	if contract := nativeOnlyDimensionContract(0); contract != nil {
		t.Fatalf("zero dimension returned a contract: %+v", contract)
	}

	contract := nativeOnlyDimensionContract(768)
	if contract == nil {
		t.Fatal("expected a native-only contract")
	}
	if got, err := contract.Resolve(0); err != nil || got != 768 {
		t.Fatalf("native dimension = %d, err=%v; want 768", got, err)
	}
	if _, err := contract.Resolve(256); err == nil {
		t.Fatal("native-only contract accepted an unsupported reduced dimension")
	}
}
