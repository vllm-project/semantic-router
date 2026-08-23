package quota

import (
	"encoding/json"
	"errors"
	"fmt"
	"strconv"
)

const (
	// QuotaIntegerDigits is the maximum number of base-10 digits in a quota
	// quantity.
	QuotaIntegerDigits = 42
	// QuotaIntegerLimbCount and QuotaIntegerLimbBase define the exact runtime
	// representation shared with quota storage adapters.
	QuotaIntegerLimbCount  = 6
	QuotaIntegerLimbBase   = uint32(10_000_000)
	quotaIntegerLimbDigits = 7
)

var (
	ErrInvalidQuotaInteger   = errors.New("invalid quota integer")
	ErrQuotaIntegerOverflow  = errors.New("quota integer overflow")
	ErrQuotaIntegerUnderflow = errors.New("quota integer underflow")
)

// QuotaInteger is a non-negative integer with at most 42 decimal digits.
// Its zero value is the number zero.
//
// Limbs are stored most-significant first so comparison never converts the
// value to a machine integer or floating-point number.
type QuotaInteger struct {
	limbs [QuotaIntegerLimbCount]uint32
}

// ParseQuotaInteger parses a canonical non-negative decimal integer.
// Canonical input has no sign, whitespace, exponent, decimal point, or leading
// zeroes, except for the value "0" itself.
func ParseQuotaInteger(value string) (QuotaInteger, error) {
	if value == "" {
		return QuotaInteger{}, fmt.Errorf("%w: value is empty", ErrInvalidQuotaInteger)
	}
	if len(value) > QuotaIntegerDigits {
		return QuotaInteger{}, fmt.Errorf(
			"%w: value exceeds %d digits",
			ErrQuotaIntegerOverflow,
			QuotaIntegerDigits,
		)
	}
	if len(value) > 1 && value[0] == '0' {
		return QuotaInteger{}, fmt.Errorf("%w: leading zeroes are not canonical", ErrInvalidQuotaInteger)
	}
	for index := range value {
		if value[index] < '0' || value[index] > '9' {
			return QuotaInteger{}, fmt.Errorf("%w: non-decimal character at offset %d", ErrInvalidQuotaInteger, index)
		}
	}

	var result QuotaInteger
	end := len(value)
	for limbIndex := QuotaIntegerLimbCount - 1; end > 0; limbIndex-- {
		start := max(0, end-quotaIntegerLimbDigits)
		limb, err := strconv.ParseUint(value[start:end], 10, 32)
		if err != nil {
			return QuotaInteger{}, fmt.Errorf("%w: %w", ErrInvalidQuotaInteger, err)
		}
		result.limbs[limbIndex] = uint32(limb)
		end = start
	}
	return result, nil
}

// NewQuotaIntegerFromLimbs validates and copies a most-significant-first limb
// representation.
func NewQuotaIntegerFromLimbs(limbs [QuotaIntegerLimbCount]uint32) (QuotaInteger, error) {
	for index, limb := range limbs {
		if limb >= QuotaIntegerLimbBase {
			return QuotaInteger{}, fmt.Errorf(
				"%w: limb %d is not less than %d",
				ErrInvalidQuotaInteger,
				index,
				QuotaIntegerLimbBase,
			)
		}
	}
	return QuotaInteger{limbs: limbs}, nil
}

// Limbs returns a copy of the most-significant-first fixed-width
// representation.
func (q QuotaInteger) Limbs() [QuotaIntegerLimbCount]uint32 {
	return q.limbs
}

func (q QuotaInteger) String() string {
	first := 0
	for first < len(q.limbs)-1 && q.limbs[first] == 0 {
		first++
	}

	result := strconv.FormatUint(uint64(q.limbs[first]), 10)
	for index := first + 1; index < len(q.limbs); index++ {
		result += fmt.Sprintf("%07d", q.limbs[index])
	}
	return result
}

// IsZero reports whether q is zero.
func (q QuotaInteger) IsZero() bool {
	return q == QuotaInteger{}
}

// Compare returns -1, 0, or 1 when q is less than, equal to, or greater than
// other.
func (q QuotaInteger) Compare(other QuotaInteger) int {
	for index := range q.limbs {
		if q.limbs[index] < other.limbs[index] {
			return -1
		}
		if q.limbs[index] > other.limbs[index] {
			return 1
		}
	}
	return 0
}

// Add returns q+other or ErrQuotaIntegerOverflow when the exact result needs
// more than 42 decimal digits.
func (q QuotaInteger) Add(other QuotaInteger) (QuotaInteger, error) {
	var result QuotaInteger
	var carry uint64
	for index := len(q.limbs) - 1; index >= 0; index-- {
		total := uint64(q.limbs[index]) + uint64(other.limbs[index]) + carry
		result.limbs[index] = uint32(total % uint64(QuotaIntegerLimbBase))
		carry = total / uint64(QuotaIntegerLimbBase)
	}
	if carry != 0 {
		return QuotaInteger{}, ErrQuotaIntegerOverflow
	}
	return result, nil
}

// Mul returns q*other or ErrQuotaIntegerOverflow when the exact product needs
// more than 42 decimal digits. The schoolbook accumulator never converts a
// quota quantity to float or a platform-sized integer.
func (q QuotaInteger) Mul(other QuotaInteger) (QuotaInteger, error) {
	var product [QuotaIntegerLimbCount * 2]uint64
	for left := len(q.limbs) - 1; left >= 0; left-- {
		for right := len(other.limbs) - 1; right >= 0; right-- {
			position := left + right + 1
			product[position] += uint64(q.limbs[left]) * uint64(other.limbs[right])
		}
	}
	base := uint64(QuotaIntegerLimbBase)
	for position := len(product) - 1; position > 0; position-- {
		product[position-1] += product[position] / base
		product[position] %= base
	}
	if product[0] >= base {
		return QuotaInteger{}, ErrQuotaIntegerOverflow
	}
	for position := 0; position < QuotaIntegerLimbCount; position++ {
		if product[position] != 0 {
			return QuotaInteger{}, ErrQuotaIntegerOverflow
		}
	}
	var result QuotaInteger
	for position := range result.limbs {
		result.limbs[position] = uint32(product[position+QuotaIntegerLimbCount])
	}
	return result, nil
}

// Sub returns q-other or ErrQuotaIntegerUnderflow when other is greater than
// q.
func (q QuotaInteger) Sub(other QuotaInteger) (QuotaInteger, error) {
	if q.Compare(other) < 0 {
		return QuotaInteger{}, ErrQuotaIntegerUnderflow
	}

	var result QuotaInteger
	var borrow int64
	for index := len(q.limbs) - 1; index >= 0; index-- {
		difference := int64(q.limbs[index]) - int64(other.limbs[index]) - borrow
		if difference < 0 {
			difference += int64(QuotaIntegerLimbBase)
			borrow = 1
		} else {
			borrow = 0
		}
		result.limbs[index] = uint32(difference)
	}
	return result, nil
}

// MarshalText emits the canonical decimal representation.
func (q QuotaInteger) MarshalText() ([]byte, error) {
	return []byte(q.String()), nil
}

// UnmarshalText accepts only the canonical decimal representation.
func (q *QuotaInteger) UnmarshalText(text []byte) error {
	if q == nil {
		return fmt.Errorf("%w: nil destination", ErrInvalidQuotaInteger)
	}
	parsed, err := ParseQuotaInteger(string(text))
	if err != nil {
		return err
	}
	*q = parsed
	return nil
}

// MarshalJSON intentionally emits a JSON string so quota quantities cannot
// lose precision in JSON-number consumers.
func (q QuotaInteger) MarshalJSON() ([]byte, error) {
	return json.Marshal(q.String())
}

// UnmarshalJSON rejects JSON numbers and accepts only a canonical decimal
// string.
func (q *QuotaInteger) UnmarshalJSON(data []byte) error {
	var value string
	if err := json.Unmarshal(data, &value); err != nil {
		return fmt.Errorf("%w: quota quantities must be JSON strings: %w", ErrInvalidQuotaInteger, err)
	}
	return q.UnmarshalText([]byte(value))
}
