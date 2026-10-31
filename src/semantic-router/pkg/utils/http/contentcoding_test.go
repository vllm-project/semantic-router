package http

import "testing"

func TestContentCoding(t *testing.T) {
	tests := []struct {
		values []string
		coding string
		single bool
	}{
		{values: nil, coding: "", single: true},
		{values: []string{""}, coding: "", single: true},
		{values: []string{"identity"}, coding: "", single: true},
		{values: []string{"gzip"}, coding: "gzip", single: true},
		{values: []string{"GZip"}, coding: "gzip", single: true},
		{values: []string{" x-gzip "}, coding: "gzip", single: true},
		{values: []string{"identity, gzip"}, coding: "gzip", single: true},
		{values: []string{"identity", "gzip"}, coding: "gzip", single: true},
		{values: []string{"ZSTD"}, coding: "zstd", single: true},
		{values: []string{"br"}, coding: "br", single: true},
		{values: []string{"gzip, br"}, single: false},
		{values: []string{"gzip", "gzip"}, single: false},
		{values: []string{"x-gzip", "gzip"}, single: false},
	}
	for _, tc := range tests {
		coding, single := ContentCoding(tc.values)
		if single != tc.single || (single && coding != tc.coding) {
			t.Errorf("ContentCoding(%q) = %q, %t; want %q, %t", tc.values, coding, single, tc.coding, tc.single)
		}
	}
}
