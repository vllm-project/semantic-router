package managementapi

import (
	"fmt"
	"regexp"
)

const (
	MinimumIdempotencyKeyLength = 16
	MaximumIdempotencyKeyLength = 200
)

var idempotencyKeyPattern = regexp.MustCompile(`^[!-~]+$`)

type IdempotencyKey string

func ParseIdempotencyKey(value string) (IdempotencyKey, error) {
	if len(value) < MinimumIdempotencyKeyLength || len(value) > MaximumIdempotencyKeyLength || !idempotencyKeyPattern.MatchString(value) {
		return "", fmt.Errorf("idempotency key must contain %d-%d visible ASCII characters", MinimumIdempotencyKeyLength, MaximumIdempotencyKeyLength)
	}
	return IdempotencyKey(value), nil
}
