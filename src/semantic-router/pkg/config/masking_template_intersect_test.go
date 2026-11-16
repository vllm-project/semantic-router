package config

import (
	"math/rand"
	"strconv"
	"strings"
	"testing"
)

func TestMaskingTemplatesIntersect(t *testing.T) {
	for _, tc := range []struct {
		a, b, written string
	}{
		{"[V{index}X0]", "[V0X{index}]", "[V0X0]"},
		{"[V0X{index}]", "[V{index}X0]", "[V0X0]"},
		{"[A{index}B12]", "[A7B{index}]", "[A7B12]"},
		{"[V{index}X01]", "[V0X{index}]", ""},
		{"[V{index}X]", "[V0X{index}]", ""},
		{"[PERSON_{index}]", "[PERSON_{index}_X]", ""},
		{"[PERSON_{index}]", "[EMAIL_{index}]", ""},
		{"[V{index}X0]", "[V01X{index}]", ""},
	} {
		written, ok := maskingTemplatesIntersect(tc.a, tc.b)
		if ok != (tc.written != "") || ok && written != tc.written {
			t.Errorf("maskingTemplatesIntersect(%q, %q) = %q, %t; want %q", tc.a, tc.b, written, ok, tc.written)
		}
	}
}

// TestMaskingTemplatesIntersectMatchesEnumeration checks the closed form
// against every placeholder the two templates write at indexes below 1000.
func TestMaskingTemplatesIntersectMatchesEnumeration(t *testing.T) {
	random := rand.New(rand.NewSource(6605))
	// Prefixes and suffixes keep a non-digit beside {index}, as validation
	// does; digits inside them are what can collide. Four characters reach
	// two-digit runs such as 01, which no index writes, and keep every
	// colliding index below 1000.
	piece := func() string {
		alphabet := "XY01"
		var b strings.Builder
		for range 1 + random.Intn(4) {
			b.WriteByte(alphabet[random.Intn(len(alphabet))])
		}
		return b.String()
	}
	digit := func(c byte) bool { return c == '0' || c == '1' }
	// closed keeps a non-digit at each end of s that sits beside {index}.
	closed := func(s string, first, last bool) string {
		if first && digit(s[0]) {
			s = "X" + s[1:]
		}
		if last && digit(s[len(s)-1]) {
			s = s[:len(s)-1] + "Y"
		}
		return s
	}
	index := func() string { return strconv.Itoa(random.Intn(12)) }
	pair := func() (string, string) {
		if random.Intn(2) == 0 {
			a := "[" + closed(piece(), false, true) + MaskingPlaceholderIndexToken + closed(piece(), true, false) + "]"
			b := "[" + closed(piece(), false, true) + MaskingPlaceholderIndexToken + closed(piece(), true, false) + "]"
			return a, b
		}
		// a = P{index}B m S and b = P n B{index}S both write P n B m S. One
		// changed character then turns many of these into near misses.
		p, mid, s := closed(piece(), false, true), closed(piece(), true, true), closed(piece(), true, false)
		a := "[" + p + MaskingPlaceholderIndexToken + mid + index() + s + "]"
		b := "[" + p + index() + mid + MaskingPlaceholderIndexToken + s + "]"
		if random.Intn(2) == 0 {
			at := 1 + random.Intn(len(b)-2)
			if !strings.Contains(MaskingPlaceholderIndexToken, b[at:at+1]) {
				b = b[:at] + string("XY01"[random.Intn(4)]) + b[at+1:]
			}
		}
		if random.Intn(2) == 0 {
			return b, a
		}
		return a, b
	}
	valid := func(template string) bool {
		before, after, _ := strings.Cut(template, MaskingPlaceholderIndexToken)
		return strings.Count(template, MaskingPlaceholderIndexToken) == 1 && !digit(before[len(before)-1]) && !digit(after[0])
	}
	const below = 1000
	write := func(template string) map[string]bool {
		out := make(map[string]bool, below)
		for i := range below {
			out[strings.Replace(template, MaskingPlaceholderIndexToken, strconv.Itoa(i), 1)] = true
		}
		return out
	}
	intersecting, disjoint := 0, 0
	for range 20000 {
		a, b := pair()
		if a == b || !valid(a) || !valid(b) {
			continue
		}
		written, ok := maskingTemplatesIntersect(a, b)
		inA, inB := write(a), write(b)
		var shared string
		for s := range inA {
			if inB[s] {
				shared = s
				break
			}
		}
		if ok != (shared != "") || ok && !(inA[written] && inB[written]) {
			t.Fatalf("maskingTemplatesIntersect(%q, %q) = %q, %t; enumeration found %q", a, b, written, ok, shared)
		}
		if ok {
			intersecting++
		} else {
			disjoint++
		}
	}
	if intersecting < 1000 || disjoint < 1000 {
		t.Fatalf("the generator produced %d intersecting and %d disjoint pairs; both need to be common for the check to prove anything", intersecting, disjoint)
	}
}

func TestRegistryMaskingRefusesIntersectingTemplates(t *testing.T) {
	plugin := func(index string, placeholders map[string]string) *MaskingPluginConfig {
		return &MaskingPluginConfig{
			Enabled:           true,
			Signal:            "personal",
			EntityTypes:       []string{"PERSON", "EMAIL_ADDRESS"},
			Index:             index,
			IdentityNamespace: map[bool]string{true: "ids"}[index == MaskingIndexRegistry],
			Placeholders:      placeholders,
		}
	}
	intersecting := map[string]string{"PERSON": "[V{index}X0]", "EMAIL_ADDRESS": "[V0X{index}]"}
	err := validateMaskingPlugin(PluginAt{}, plugin(MaskingIndexRegistry, intersecting))
	if err == nil || !strings.Contains(err.Error(), `"EMAIL_ADDRESS" and "PERSON" can both produce [V0X0]`) {
		t.Fatalf("registry error = %v, want the intersecting templates refused", err)
	}
	if err := validateMaskingPlugin(PluginAt{}, plugin(MaskingIndexSequence, intersecting)); err != nil {
		t.Fatalf("sequence mode refused templates its allocator keeps apart: %v", err)
	}
	disjoint := map[string]string{"PERSON": "[P{index}]", "EMAIL_ADDRESS": "[E{index}]"}
	if err := validateMaskingPlugin(PluginAt{}, plugin(MaskingIndexRegistry, disjoint)); err != nil {
		t.Fatalf("registry mode refused disjoint templates: %v", err)
	}
}
