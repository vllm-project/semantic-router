package cache

import (
	"testing"
	"time"
)

func TestExactLookupResultFromTimestampsHonorsExpiration(t *testing.T) {
	now := time.Now()
	tests := []struct {
		name      string
		expiresAt time.Time
		wantFound bool
	}{
		{name: "future expiry", expiresAt: now.Add(time.Minute), wantFound: true},
		{name: "expired", expiresAt: now.Add(-time.Minute), wantFound: false},
		{name: "no expiry", expiresAt: time.Time{}, wantFound: true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := exactLookupResultFromTimestamps([]byte("cached"), now, tt.expiresAt)
			if result.Found != tt.wantFound {
				t.Fatalf("Found = %v, want %v; result = %#v", result.Found, tt.wantFound, result)
			}
			if tt.wantFound && string(result.ResponseBody) != "cached" {
				t.Fatalf("ResponseBody = %q, want cached", result.ResponseBody)
			}
		})
	}
}
