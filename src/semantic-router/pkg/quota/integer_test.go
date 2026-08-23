package quota

import (
	"encoding/json"
	"errors"
	"strings"
	"testing"
)

func TestParseQuotaInteger(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name  string
		input string
		want  string
	}{
		{name: "zero", input: "0", want: "0"},
		{name: "single limb", input: "9999999", want: "9999999"},
		{name: "limb boundary", input: "10000000", want: "10000000"},
		{name: "maximum", input: strings.Repeat("9", QuotaIntegerDigits), want: strings.Repeat("9", QuotaIntegerDigits)},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			got, err := ParseQuotaInteger(test.input)
			if err != nil {
				t.Fatalf("ParseQuotaInteger(%q) error = %v", test.input, err)
			}
			if got.String() != test.want {
				t.Fatalf("ParseQuotaInteger(%q).String() = %q, want %q", test.input, got.String(), test.want)
			}
		})
	}
}

func TestParseQuotaIntegerRejectsNonCanonicalInput(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name    string
		input   string
		wantErr error
	}{
		{name: "empty", input: "", wantErr: ErrInvalidQuotaInteger},
		{name: "leading zero", input: "01", wantErr: ErrInvalidQuotaInteger},
		{name: "plus sign", input: "+1", wantErr: ErrInvalidQuotaInteger},
		{name: "negative", input: "-1", wantErr: ErrInvalidQuotaInteger},
		{name: "decimal", input: "1.0", wantErr: ErrInvalidQuotaInteger},
		{name: "exponent", input: "1e3", wantErr: ErrInvalidQuotaInteger},
		{name: "whitespace", input: " 1", wantErr: ErrInvalidQuotaInteger},
		{name: "too many digits", input: "1" + strings.Repeat("0", QuotaIntegerDigits), wantErr: ErrQuotaIntegerOverflow},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			_, err := ParseQuotaInteger(test.input)
			if !errors.Is(err, test.wantErr) {
				t.Fatalf("ParseQuotaInteger(%q) error = %v, want %v", test.input, err, test.wantErr)
			}
		})
	}
}

func TestQuotaIntegerAdd(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name    string
		left    string
		right   string
		want    string
		wantErr error
	}{
		{name: "zero", left: "0", right: "0", want: "0"},
		{name: "simple", left: "12", right: "30", want: "42"},
		{name: "carry one limb", left: "9999999", right: "1", want: "10000000"},
		{name: "carry every limb", left: strings.Repeat("9", 41), right: "1", want: "1" + strings.Repeat("0", 41)},
		{name: "overflow", left: strings.Repeat("9", QuotaIntegerDigits), right: "1", wantErr: ErrQuotaIntegerOverflow},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			got, err := mustQuotaInteger(t, test.left).Add(mustQuotaInteger(t, test.right))
			if !errors.Is(err, test.wantErr) {
				t.Fatalf("Add() error = %v, want %v", err, test.wantErr)
			}
			if test.wantErr == nil && got.String() != test.want {
				t.Fatalf("Add() = %q, want %q", got.String(), test.want)
			}
		})
	}
}

func TestQuotaIntegerMultiply(t *testing.T) {
	t.Parallel()

	tests := []struct {
		left  string
		right string
		want  string
	}{
		{left: "0", right: "999999999999999999", want: "0"},
		{left: "1", right: "999999999999999999", want: "999999999999999999"},
		{left: "123456789", right: "987654321", want: "121932631112635269"},
		{left: "999999999999999999999", right: "999999999999999999999", want: "999999999999999999998000000000000000000001"},
	}
	for _, test := range tests {
		left := mustQuotaInteger(t, test.left)
		right := mustQuotaInteger(t, test.right)
		got, err := left.Mul(right)
		if err != nil {
			t.Fatalf("Mul(%s, %s) error = %v", test.left, test.right, err)
		}
		if got.String() != test.want {
			t.Fatalf("Mul(%s, %s) = %s, want %s", test.left, test.right, got.String(), test.want)
		}
	}
}

func TestQuotaIntegerMultiplyOverflow(t *testing.T) {
	t.Parallel()

	maximum := mustQuotaInteger(t, strings.Repeat("9", QuotaIntegerDigits))
	two := mustQuotaInteger(t, "2")
	if _, err := maximum.Mul(two); !errors.Is(err, ErrQuotaIntegerOverflow) {
		t.Fatalf("Mul() overflow error = %v, want %v", err, ErrQuotaIntegerOverflow)
	}
}

func TestQuotaIntegerSub(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name    string
		left    string
		right   string
		want    string
		wantErr error
	}{
		{name: "zero", left: "0", right: "0", want: "0"},
		{name: "simple", left: "42", right: "12", want: "30"},
		{name: "borrow one limb", left: "10000000", right: "1", want: "9999999"},
		{name: "borrow every limb", left: "1" + strings.Repeat("0", 41), right: "1", want: strings.Repeat("9", 41)},
		{name: "underflow", left: "1", right: "2", wantErr: ErrQuotaIntegerUnderflow},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			got, err := mustQuotaInteger(t, test.left).Sub(mustQuotaInteger(t, test.right))
			if !errors.Is(err, test.wantErr) {
				t.Fatalf("Sub() error = %v, want %v", err, test.wantErr)
			}
			if test.wantErr == nil && got.String() != test.want {
				t.Fatalf("Sub() = %q, want %q", got.String(), test.want)
			}
		})
	}
}

func TestQuotaIntegerCompare(t *testing.T) {
	t.Parallel()

	tests := []struct {
		left  string
		right string
		want  int
	}{
		{left: "0", right: "0", want: 0},
		{left: "9999999", right: "10000000", want: -1},
		{left: "10000000", right: "9999999", want: 1},
		{left: strings.Repeat("9", 42), right: strings.Repeat("9", 42), want: 0},
	}
	for _, test := range tests {
		if got := mustQuotaInteger(t, test.left).Compare(mustQuotaInteger(t, test.right)); got != test.want {
			t.Errorf("Compare(%s, %s) = %d, want %d", test.left, test.right, got, test.want)
		}
	}
}

func TestQuotaIntegerLimbRoundTrip(t *testing.T) {
	t.Parallel()

	want := mustQuotaInteger(t, "123456789012345678901234567890123456789012")
	got, err := NewQuotaIntegerFromLimbs(want.Limbs())
	if err != nil {
		t.Fatalf("NewQuotaIntegerFromLimbs() error = %v", err)
	}
	if got != want {
		t.Fatalf("limb round trip = %q, want %q", got.String(), want.String())
	}

	invalid := [QuotaIntegerLimbCount]uint32{QuotaIntegerLimbBase}
	if _, err := NewQuotaIntegerFromLimbs(invalid); !errors.Is(err, ErrInvalidQuotaInteger) {
		t.Fatalf("invalid limb error = %v, want %v", err, ErrInvalidQuotaInteger)
	}
}

func TestQuotaIntegerJSONRequiresString(t *testing.T) {
	t.Parallel()

	value := mustQuotaInteger(t, "12345678901234567890")
	encoded, err := json.Marshal(value)
	if err != nil {
		t.Fatalf("json.Marshal() error = %v", err)
	}
	if string(encoded) != `"12345678901234567890"` {
		t.Fatalf("json.Marshal() = %s", encoded)
	}

	var decoded QuotaInteger
	if err := json.Unmarshal(encoded, &decoded); err != nil {
		t.Fatalf("json.Unmarshal(string) error = %v", err)
	}
	if decoded != value {
		t.Fatalf("json round trip = %q, want %q", decoded.String(), value.String())
	}
	if err := json.Unmarshal([]byte(`123`), &decoded); !errors.Is(err, ErrInvalidQuotaInteger) {
		t.Fatalf("json.Unmarshal(number) error = %v, want %v", err, ErrInvalidQuotaInteger)
	}
}

func mustQuotaInteger(t *testing.T, value string) QuotaInteger {
	t.Helper()
	parsed, err := ParseQuotaInteger(value)
	if err != nil {
		t.Fatalf("ParseQuotaInteger(%q) error = %v", value, err)
	}
	return parsed
}
