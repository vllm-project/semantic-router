package quota

import (
	"encoding/json"
	"errors"
	"strings"
	"testing"
)

func TestParseCurrencyDecimal(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name   string
		input  string
		string string
		scaled string
	}{
		{name: "zero", input: "0", string: "0", scaled: "0"},
		{name: "integer", input: "5", string: "5", scaled: "5000000000000000"},
		{name: "fraction", input: "2.5", string: "2.5", scaled: "2500000000000000"},
		{name: "minimum fraction", input: "0.000000000000001", string: "0.000000000000001", scaled: "1"},
		{name: "maximum", input: strings.Repeat("9", 27) + "." + strings.Repeat("9", 15), string: strings.Repeat("9", 27) + "." + strings.Repeat("9", 15), scaled: strings.Repeat("9", 42)},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			got, err := ParseCurrencyDecimal(test.input)
			if err != nil {
				t.Fatalf("ParseCurrencyDecimal(%q) error = %v", test.input, err)
			}
			if got.String() != test.string {
				t.Errorf("String() = %q, want %q", got.String(), test.string)
			}
			if got.ScaledInteger().String() != test.scaled {
				t.Errorf("ScaledInteger() = %q, want %q", got.ScaledInteger().String(), test.scaled)
			}
		})
	}
}

func TestParseCurrencyDecimalRejectsNonCanonicalInput(t *testing.T) {
	t.Parallel()

	tests := []string{
		"",
		"00",
		"01",
		"1.",
		".1",
		"1.0",
		"1.20",
		"+1",
		"-1",
		"1e3",
		" 1",
		"1..2",
		"0.0000000000000001",
		strings.Repeat("9", 28),
	}
	for _, input := range tests {
		t.Run(input, func(t *testing.T) {
			t.Parallel()
			if _, err := ParseCurrencyDecimal(input); !errors.Is(err, ErrInvalidCurrencyDecimal) {
				t.Fatalf("ParseCurrencyDecimal(%q) error = %v, want %v", input, err, ErrInvalidCurrencyDecimal)
			}
		})
	}
}

func TestCurrencyDecimalExactArithmetic(t *testing.T) {
	t.Parallel()

	left := mustCurrencyDecimal(t, "1.000000000000001")
	right := mustCurrencyDecimal(t, "2.000000000000009")
	sum, err := left.Add(right)
	if err != nil {
		t.Fatalf("Add() error = %v", err)
	}
	if sum.String() != "3.00000000000001" {
		t.Fatalf("Add() = %q, want %q", sum.String(), "3.00000000000001")
	}
	difference, err := sum.Sub(right)
	if err != nil {
		t.Fatalf("Sub() error = %v", err)
	}
	if difference != left {
		t.Fatalf("Sub() = %q, want %q", difference.String(), left.String())
	}
	if _, err := left.Sub(right); !errors.Is(err, ErrQuotaIntegerUnderflow) {
		t.Fatalf("underflow error = %v, want %v", err, ErrQuotaIntegerUnderflow)
	}
	equal := mustCurrencyDecimal(t, left.String())
	if left.Compare(right) >= 0 || right.Compare(left) <= 0 || left.Compare(equal) != 0 {
		t.Fatal("Compare() returned inconsistent ordering")
	}
}

func TestCurrencyDecimalOverflow(t *testing.T) {
	t.Parallel()

	maxValue := mustCurrencyDecimal(t, strings.Repeat("9", 27)+"."+strings.Repeat("9", 15))
	minimum := mustCurrencyDecimal(t, "0.000000000000001")
	if _, err := maxValue.Add(minimum); !errors.Is(err, ErrQuotaIntegerOverflow) {
		t.Fatalf("Add() overflow error = %v, want %v", err, ErrQuotaIntegerOverflow)
	}
}

func TestCurrencyDecimalJSONRequiresString(t *testing.T) {
	t.Parallel()

	value := mustCurrencyDecimal(t, "2.5")
	encoded, err := json.Marshal(value)
	if err != nil {
		t.Fatalf("json.Marshal() error = %v", err)
	}
	if string(encoded) != `"2.5"` {
		t.Fatalf("json.Marshal() = %s", encoded)
	}

	var decoded CurrencyDecimal
	if err := json.Unmarshal(encoded, &decoded); err != nil {
		t.Fatalf("json.Unmarshal(string) error = %v", err)
	}
	if decoded != value {
		t.Fatalf("json round trip = %q, want %q", decoded.String(), value.String())
	}
	if err := json.Unmarshal([]byte(`2.5`), &decoded); !errors.Is(err, ErrInvalidCurrencyDecimal) {
		t.Fatalf("json.Unmarshal(number) error = %v, want %v", err, ErrInvalidCurrencyDecimal)
	}
}

func mustCurrencyDecimal(t *testing.T, value string) CurrencyDecimal {
	t.Helper()
	parsed, err := ParseCurrencyDecimal(value)
	if err != nil {
		t.Fatalf("ParseCurrencyDecimal(%q) error = %v", value, err)
	}
	return parsed
}
