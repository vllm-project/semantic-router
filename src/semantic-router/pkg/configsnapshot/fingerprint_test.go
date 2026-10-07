package configsnapshot

import "testing"

type fingerprintSample struct {
	Name    string
	hidden  int
	Tags    []string
	Labels  map[string]string
	Nested  map[interface{}]interface{}
	Pointer *int
	Any     any
}

func TestFingerprintCoversContentNotConstruction(t *testing.T) {
	one := 1
	base := fingerprintSample{
		Name: "a", hidden: 1, Tags: []string{"x"},
		Labels:  map[string]string{"k1": "v1", "k2": "v2"},
		Nested:  map[interface{}]interface{}{"inner": map[interface{}]interface{}{"deep": 1}},
		Pointer: &one, Any: 3,
	}
	reordered := base
	reordered.Labels = map[string]string{"k2": "v2", "k1": "v1"}
	if fingerprint(base) != fingerprint(reordered) {
		t.Fatal("map insertion order changed the fingerprint")
	}
	otherOne := 1
	samePointee := base
	samePointee.Pointer = &otherOne
	if fingerprint(base) != fingerprint(samePointee) {
		t.Fatal("an equal pointee behind another pointer changed the fingerprint")
	}

	for name, change := range map[string]func(*fingerprintSample){
		"exported field":    func(s *fingerprintSample) { s.Name = "b" },
		"unexported field":  func(s *fingerprintSample) { s.hidden = 2 },
		"nil slice":         func(s *fingerprintSample) { s.Tags = nil },
		"empty slice":       func(s *fingerprintSample) { s.Tags = []string{} },
		"map value":         func(s *fingerprintSample) { s.Labels = map[string]string{"k1": "v1", "k2": "v3"} },
		"nested yaml value": func(s *fingerprintSample) { s.Nested = map[interface{}]interface{}{"inner": 2} },
		"nil pointer":       func(s *fingerprintSample) { s.Pointer = nil },
		"interface type":    func(s *fingerprintSample) { s.Any = int64(3) },
	} {
		changed := base
		change(&changed)
		if fingerprint(base) == fingerprint(changed) {
			t.Errorf("%s: the fingerprint did not change", name)
		}
	}
	if fingerprint("ab", "c") == fingerprint("a", "bc") {
		t.Fatal("adjacent values run together")
	}
}

func TestSecretFingerprintIsKeyed(t *testing.T) {
	first, second := secretFingerprint("value"), secretFingerprint("value")
	if first == fingerprint("value") {
		t.Fatal("a secret fingerprint equals the unkeyed digest of its value")
	}
	if first != second {
		t.Fatal("a secret fingerprint is unstable within the process")
	}
}

type cycle struct{ Next *cycle }

func TestFingerprintTerminatesOnACycle(t *testing.T) {
	loop := &cycle{}
	loop.Next = loop
	if fingerprint(loop) == "" {
		t.Fatal("no fingerprint for a cyclic value")
	}
}
