package policymanagement

import (
	"encoding/json"
	"errors"
	"testing"
	"time"
)

func TestISODurationJSONRoundTrip(t *testing.T) {
	tests := []struct {
		duration time.Duration
		encoded  string
	}{
		{0, `"PT0S"`},
		{time.Minute, `"PT1M"`},
		{24*time.Hour + 2*time.Hour + 3*time.Minute + 4*time.Second, `"P1DT2H3M4S"`},
		{time.Millisecond, `"PT0.001S"`},
		{time.Microsecond, `"PT0.000001S"`},
	}
	for _, test := range tests {
		encoded, err := json.Marshal(ISODuration(test.duration))
		if err != nil || string(encoded) != test.encoded {
			t.Fatalf("marshal %s = %s, %v", test.duration, encoded, err)
		}
		var decoded ISODuration
		if err := json.Unmarshal(encoded, &decoded); err != nil || decoded.Duration() != test.duration {
			t.Fatalf("unmarshal %s = %s, %v", encoded, decoded.Duration(), err)
		}
	}
}

func TestISODurationRejectsNonCanonicalOrUnsafeValues(t *testing.T) {
	for _, encoded := range []string{
		`60000000000`, `null`, `""`, `"P"`, `"PT"`, `"PT60S"`, `"PT01M"`,
		`"-PT1M"`, `"P999999999999999999999D"`, `"P0D"`, `"PT1.000S"`,
	} {
		var duration ISODuration
		if err := json.Unmarshal([]byte(encoded), &duration); err == nil {
			t.Fatalf("accepted non-canonical duration %s", encoded)
		}
	}
	if _, err := json.Marshal(ISODuration(-time.Second)); !errors.Is(err, ErrInvalidRequest) {
		t.Fatalf("negative duration error = %v", err)
	}
}

func TestRateLimitRuleJSONUsesISO8601Duration(t *testing.T) {
	rule := RateLimitRule{
		Window:               ISODuration(time.Minute),
		RefillPeriod:         ISODuration(time.Millisecond),
		GCRAEmissionInterval: ISODuration(time.Microsecond),
	}
	encoded, err := json.Marshal(rule)
	if err != nil {
		t.Fatal(err)
	}
	if string(encoded) != `{"metric":"","algorithm":"","window":"PT1M","refillPeriod":"PT0.001S","emissionInterval":"PT0.000001S","accounting":"","enforcement":"","ordinal":0}` {
		t.Fatalf("rate rule JSON = %s", encoded)
	}
}
