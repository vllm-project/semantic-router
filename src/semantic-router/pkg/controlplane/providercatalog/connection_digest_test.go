package providercatalog

import "testing"

func TestCanonicalConnectionDigestIgnoresMapOrderAndExplicitDefaults(t *testing.T) {
	schema := []ConnectionField{
		{Name: "region", Kind: FieldText, Default: "global"},
		{Name: "batch", Kind: FieldInteger, Required: true},
	}
	implicit, err := normalizeConnectionFields(schema, map[string]any{"batch": 4})
	if err != nil {
		t.Fatal(err)
	}
	explicit, err := normalizeConnectionFields(schema, map[string]any{"region": "global", "batch": 4})
	if err != nil {
		t.Fatal(err)
	}
	first, err := CanonicalConnectionDigest(implicit)
	if err != nil {
		t.Fatal(err)
	}
	second, err := CanonicalConnectionDigest(explicit)
	if err != nil {
		t.Fatal(err)
	}
	if first != second {
		t.Fatalf("equivalent connection forms differ: %s != %s", first, second)
	}
	changed, err := CanonicalConnectionDigest(map[string]CanonicalConnectionValue{
		"batch":  {Kind: FieldInteger, Value: "4"},
		"region": {Kind: FieldText, Value: "eu"},
	})
	if err != nil {
		t.Fatal(err)
	}
	if first == changed {
		t.Fatal("semantic connection change retained the same digest")
	}
}
