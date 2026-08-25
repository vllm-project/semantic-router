package routingmanagement

import (
	"errors"
	"testing"
	"time"
)

func TestResolveModelProbeTimeout(t *testing.T) {
	tests := []struct {
		name       string
		configured string
		explicit   time.Duration
		want       time.Duration
		wantErr    error
	}{
		{name: "saved request timeout", configured: "45s", want: 45 * time.Second},
		{name: "saved timeout capped for probe", configured: "10m", want: 5 * time.Minute},
		{name: "explicit override", configured: "10m", explicit: 30 * time.Second, want: 30 * time.Second},
		{name: "invalid saved timeout", configured: "invalid", wantErr: ErrProbeUnavailable},
		{name: "explicit timeout too short", configured: "45s", explicit: time.Millisecond, wantErr: ErrInvalid},
		{name: "explicit timeout too long", configured: "45s", explicit: 6 * time.Minute, wantErr: ErrInvalid},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			got, err := resolveModelProbeTimeout(test.configured, test.explicit)
			if test.wantErr != nil {
				if !errors.Is(err, test.wantErr) {
					t.Fatalf("resolveModelProbeTimeout() error = %v, want %v", err, test.wantErr)
				}
				return
			}
			if err != nil || got != test.want {
				t.Fatalf("resolveModelProbeTimeout() = %v, %v; want %v, nil", got, err, test.want)
			}
		})
	}
}
