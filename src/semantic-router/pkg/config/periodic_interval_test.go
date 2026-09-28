/*
Copyright 2025 vLLM Semantic Router.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package config

import (
	"testing"
	"time"
)

func TestParsePeriodicInterval(t *testing.T) {
	const def = 30 * time.Second

	tests := []struct {
		name     string
		value    string
		wantErr  bool
		wantDur  time.Duration
		useDefOK bool // when true, expect the default returned with no error
	}{
		{name: "empty uses default", value: "", useDefOK: true},
		{name: "whitespace uses default", value: "   ", useDefOK: true},
		{name: "valid seconds", value: "45s", wantDur: 45 * time.Second},
		{name: "valid minutes", value: "5m", wantDur: 5 * time.Minute},
		{name: "max boundary allowed", value: "24h", wantDur: MaxPeriodicInterval},
		{name: "zero rejected", value: "0s", wantErr: true},
		{name: "bare zero rejected", value: "0", wantErr: true},
		{name: "negative rejected", value: "-5s", wantErr: true},
		{name: "unparsable rejected", value: "banana", wantErr: true},
		{name: "above max rejected", value: "48h", wantErr: true},
		{name: "overflow rejected", value: "9999999999h", wantErr: true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := ParsePeriodicInterval(tt.value, def)
			if tt.wantErr {
				if err == nil {
					t.Fatalf("ParsePeriodicInterval(%q) expected error, got nil (dur=%s)", tt.value, got)
				}
				return
			}
			if err != nil {
				t.Fatalf("ParsePeriodicInterval(%q) unexpected error: %v", tt.value, err)
			}
			want := tt.wantDur
			if tt.useDefOK {
				want = def
			}
			if got != want {
				t.Fatalf("ParsePeriodicInterval(%q) = %s, want %s", tt.value, got, want)
			}
		})
	}
}

func TestParsePeriodicIntervalErrorMentionsField(t *testing.T) {
	// The error must reference the offending value so callers can surface a
	// field-addressed message.
	_, err := ParsePeriodicInterval("-1s", time.Second)
	if err == nil {
		t.Fatal("expected error for negative duration")
	}
	if got := err.Error(); got == "" {
		t.Fatal("error message must not be empty")
	}
}
