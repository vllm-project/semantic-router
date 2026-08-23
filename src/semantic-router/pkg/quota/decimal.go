package quota

import (
	"encoding/json"
	"errors"
	"fmt"
	"strings"
)

const CurrencyDecimalScale = 15

var ErrInvalidCurrencyDecimal = errors.New("invalid currency decimal")

// CurrencyDecimal is a canonical non-negative currency quantity with at most
// 15 fractional digits. Internally it is an exact integer scaled by 10^15.
// Its zero value is the number zero.
type CurrencyDecimal struct {
	scaled QuotaInteger
}

// ParseCurrencyDecimal parses a canonical decimal and verifies that its exact
// 10^15-scaled value fits QuotaInteger.
func ParseCurrencyDecimal(value string) (CurrencyDecimal, error) {
	if value == "" {
		return CurrencyDecimal{}, fmt.Errorf("%w: value is empty", ErrInvalidCurrencyDecimal)
	}
	if strings.Count(value, ".") > 1 {
		return CurrencyDecimal{}, fmt.Errorf("%w: multiple decimal points", ErrInvalidCurrencyDecimal)
	}

	integerPart, fractionalPart, hasPoint := strings.Cut(value, ".")
	if integerPart == "" || (hasPoint && fractionalPart == "") {
		return CurrencyDecimal{}, fmt.Errorf("%w: integer and fractional parts must be explicit", ErrInvalidCurrencyDecimal)
	}
	if !canonicalDecimalDigits(integerPart) {
		return CurrencyDecimal{}, fmt.Errorf("%w: invalid integer part", ErrInvalidCurrencyDecimal)
	}
	if len(integerPart) > 1 && integerPart[0] == '0' {
		return CurrencyDecimal{}, fmt.Errorf("%w: leading zeroes are not canonical", ErrInvalidCurrencyDecimal)
	}
	if hasPoint {
		if !canonicalDecimalDigits(fractionalPart) {
			return CurrencyDecimal{}, fmt.Errorf("%w: invalid fractional part", ErrInvalidCurrencyDecimal)
		}
		if len(fractionalPart) > CurrencyDecimalScale {
			return CurrencyDecimal{}, fmt.Errorf(
				"%w: more than %d fractional digits",
				ErrInvalidCurrencyDecimal,
				CurrencyDecimalScale,
			)
		}
		if fractionalPart[len(fractionalPart)-1] == '0' {
			return CurrencyDecimal{}, fmt.Errorf("%w: trailing zeroes are not canonical", ErrInvalidCurrencyDecimal)
		}
	}

	scaledText := integerPart
	if hasPoint {
		scaledText += fractionalPart
	}
	scaledText += strings.Repeat("0", CurrencyDecimalScale-len(fractionalPart))
	scaledText = strings.TrimLeft(scaledText, "0")
	if scaledText == "" {
		scaledText = "0"
	}
	scaled, err := ParseQuotaInteger(scaledText)
	if err != nil {
		if errors.Is(err, ErrQuotaIntegerOverflow) {
			return CurrencyDecimal{}, fmt.Errorf("%w: scaled value exceeds %d digits", ErrInvalidCurrencyDecimal, QuotaIntegerDigits)
		}
		return CurrencyDecimal{}, fmt.Errorf("%w: %w", ErrInvalidCurrencyDecimal, err)
	}
	return CurrencyDecimal{scaled: scaled}, nil
}

// NewCurrencyDecimalFromScaled creates a public currency decimal from an exact
// 10^15-scaled runtime quantity.
func NewCurrencyDecimalFromScaled(scaled QuotaInteger) CurrencyDecimal {
	return CurrencyDecimal{scaled: scaled}
}

// ScaledInteger returns the exact 10^15-scaled quantity.
func (d CurrencyDecimal) ScaledInteger() QuotaInteger {
	return d.scaled
}

func (d CurrencyDecimal) String() string {
	scaled := d.scaled.String()
	if d.scaled.IsZero() {
		return "0"
	}
	if len(scaled) <= CurrencyDecimalScale {
		scaled = strings.Repeat("0", CurrencyDecimalScale+1-len(scaled)) + scaled
	}
	point := len(scaled) - CurrencyDecimalScale
	integerPart := scaled[:point]
	fractionalPart := strings.TrimRight(scaled[point:], "0")
	if fractionalPart == "" {
		return integerPart
	}
	return integerPart + "." + fractionalPart
}

// IsZero reports whether d is zero.
func (d CurrencyDecimal) IsZero() bool {
	return d.scaled.IsZero()
}

// Compare returns -1, 0, or 1 when d is less than, equal to, or greater than
// other.
func (d CurrencyDecimal) Compare(other CurrencyDecimal) int {
	return d.scaled.Compare(other.scaled)
}

// Add returns the exact sum or ErrQuotaIntegerOverflow when its scaled value
// exceeds 42 digits.
func (d CurrencyDecimal) Add(other CurrencyDecimal) (CurrencyDecimal, error) {
	sum, err := d.scaled.Add(other.scaled)
	if err != nil {
		return CurrencyDecimal{}, err
	}
	return NewCurrencyDecimalFromScaled(sum), nil
}

// Sub returns the exact difference or ErrQuotaIntegerUnderflow.
func (d CurrencyDecimal) Sub(other CurrencyDecimal) (CurrencyDecimal, error) {
	difference, err := d.scaled.Sub(other.scaled)
	if err != nil {
		return CurrencyDecimal{}, err
	}
	return NewCurrencyDecimalFromScaled(difference), nil
}

func (d CurrencyDecimal) MarshalText() ([]byte, error) {
	return []byte(d.String()), nil
}

func (d *CurrencyDecimal) UnmarshalText(text []byte) error {
	if d == nil {
		return fmt.Errorf("%w: nil destination", ErrInvalidCurrencyDecimal)
	}
	parsed, err := ParseCurrencyDecimal(string(text))
	if err != nil {
		return err
	}
	*d = parsed
	return nil
}

func (d CurrencyDecimal) MarshalJSON() ([]byte, error) {
	return json.Marshal(d.String())
}

func (d *CurrencyDecimal) UnmarshalJSON(data []byte) error {
	var value string
	if err := json.Unmarshal(data, &value); err != nil {
		return fmt.Errorf("%w: currency quantities must be JSON strings: %w", ErrInvalidCurrencyDecimal, err)
	}
	return d.UnmarshalText([]byte(value))
}

func canonicalDecimalDigits(value string) bool {
	if value == "" {
		return false
	}
	for index := range value {
		if value[index] < '0' || value[index] > '9' {
			return false
		}
	}
	return true
}
