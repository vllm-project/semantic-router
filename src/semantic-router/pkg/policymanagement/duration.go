package policymanagement

import (
	"encoding/json"
	"fmt"
	"math"
	"regexp"
	"strconv"
	"strings"
	"time"
)

// ISODuration is the public JSON duration contract for rate-limit rules. JSON
// uses canonical ISO-8601 day/time values (for example PT1M or PT0.001S), never
// Go's nanosecond integer representation. Calendar months remain a separate
// RateLimitRule field because their length is not a fixed duration.
type ISODuration time.Duration

var isoDurationPattern = regexp.MustCompile(
	`^P(?:(0|[1-9][0-9]*)D)?(?:T(?:(0|[1-9][0-9]*)H)?(?:(0|[1-9][0-9]*)M)?(?:(0|[1-9][0-9]*)(?:\.([0-9]{1,9}))?S)?)?$`,
)

func (duration ISODuration) Duration() time.Duration { return time.Duration(duration) }

func (duration ISODuration) String() string {
	value, err := formatISODuration(time.Duration(duration))
	if err != nil {
		return ""
	}
	return value
}

func (duration ISODuration) MarshalJSON() ([]byte, error) {
	value, err := formatISODuration(time.Duration(duration))
	if err != nil {
		return nil, err
	}
	return json.Marshal(value)
}

func (duration *ISODuration) UnmarshalJSON(encoded []byte) error {
	if duration == nil {
		return ErrInvalidRequest
	}
	var value string
	if err := json.Unmarshal(encoded, &value); err != nil {
		return fmt.Errorf("decode ISO-8601 duration: %w", err)
	}
	parsed, err := parseISODuration(value)
	if err != nil {
		return err
	}
	*duration = ISODuration(parsed)
	return nil
}

func parseISODuration(value string) (time.Duration, error) {
	matches := isoDurationPattern.FindStringSubmatch(value)
	if matches == nil || (matches[1] == "" && matches[2] == "" && matches[3] == "" && matches[4] == "") ||
		(strings.Contains(value, "T") && matches[2] == "" && matches[3] == "" && matches[4] == "") {
		return 0, ErrInvalidRequest
	}
	components := []struct {
		value string
		unit  time.Duration
	}{{matches[1], 24 * time.Hour}, {matches[2], time.Hour}, {matches[3], time.Minute}, {matches[4], time.Second}}
	var total int64
	for _, component := range components {
		if component.value == "" {
			continue
		}
		parsed, err := strconv.ParseUint(component.value, 10, 64)
		if err != nil || parsed > uint64(math.MaxInt64/int64(component.unit)) {
			return 0, ErrInvalidRequest
		}
		part := int64(parsed) * int64(component.unit)
		if part > math.MaxInt64-total {
			return 0, ErrInvalidRequest
		}
		total += part
	}
	if matches[5] != "" {
		fraction := matches[5] + strings.Repeat("0", 9-len(matches[5]))
		nanoseconds, err := strconv.ParseInt(fraction, 10, 64)
		if err != nil || nanoseconds > math.MaxInt64-total {
			return 0, ErrInvalidRequest
		}
		total += nanoseconds
	}
	parsed := time.Duration(total)
	canonical, err := formatISODuration(parsed)
	if err != nil || canonical != value {
		return 0, ErrInvalidRequest
	}
	return parsed, nil
}

func formatISODuration(duration time.Duration) (string, error) {
	if duration < 0 {
		return "", ErrInvalidRequest
	}
	if duration == 0 {
		return "PT0S", nil
	}
	remaining := duration
	days := remaining / (24 * time.Hour)
	remaining %= 24 * time.Hour
	hours := remaining / time.Hour
	remaining %= time.Hour
	minutes := remaining / time.Minute
	remaining %= time.Minute
	seconds := remaining / time.Second
	nanoseconds := remaining % time.Second

	var result strings.Builder
	result.WriteByte('P')
	if days > 0 {
		result.WriteString(strconv.FormatInt(int64(days), 10))
		result.WriteByte('D')
	}
	if hours > 0 || minutes > 0 || seconds > 0 || nanoseconds > 0 {
		result.WriteByte('T')
		if hours > 0 {
			result.WriteString(strconv.FormatInt(int64(hours), 10))
			result.WriteByte('H')
		}
		if minutes > 0 {
			result.WriteString(strconv.FormatInt(int64(minutes), 10))
			result.WriteByte('M')
		}
		if seconds > 0 || nanoseconds > 0 {
			result.WriteString(strconv.FormatInt(int64(seconds), 10))
			if nanoseconds > 0 {
				fraction := fmt.Sprintf("%09d", nanoseconds)
				result.WriteByte('.')
				result.WriteString(strings.TrimRight(fraction, "0"))
			}
			result.WriteByte('S')
		}
	}
	return result.String(), nil
}
