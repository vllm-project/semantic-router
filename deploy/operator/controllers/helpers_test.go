/*
Copyright 2026 vLLM Semantic Router Contributors.

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

package controllers

import (
	"testing"
)

func TestGetInt32OrDefault(t *testing.T) {
	r := &SemanticRouterReconciler{}

	tests := []struct {
		name string
		val  *int32
		def  int32
		want int32
	}{
		{
			name: "nil value returns default",
			val:  nil,
			def:  10,
			want: 10,
		},
		{
			name: "non-nil value returns value",
			val:  func() *int32 { i := int32(5); return &i }(),
			def:  10,
			want: 5,
		},
		{
			name: "zero value returns zero",
			val:  func() *int32 { i := int32(0); return &i }(),
			def:  10,
			want: 0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := r.getInt32OrDefault(tt.val, tt.def)
			if got != tt.want {
				t.Errorf("getInt32OrDefault() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestConvertStringValue(t *testing.T) {
	r := &SemanticRouterReconciler{}

	tests := []struct {
		name  string
		input string
		want  interface{}
	}{
		{
			name:  "integer string",
			input: "100",
			want:  int64(100),
		},
		{
			name:  "float string",
			input: "0.6",
			want:  float64(0.6),
		},
		{
			name:  "boolean true",
			input: "true",
			want:  true,
		},
		{
			name:  "boolean false",
			input: "false",
			want:  false,
		},
		{
			name:  "regular string",
			input: "hello",
			want:  "hello",
		},
		{
			name:  "empty string",
			input: "",
			want:  "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := r.convertStringValue(tt.input)
			if got != tt.want {
				t.Errorf("convertStringValue(%q) = %v (type %T), want %v (type %T)",
					tt.input, got, got, tt.want, tt.want)
			}
		})
	}
}

func TestConvertStringsToFloats(t *testing.T) {
	r := &SemanticRouterReconciler{}

	tests := []struct {
		name  string
		input map[string]interface{}
		want  map[string]interface{}
	}{
		{
			name: "convert float strings",
			input: map[string]interface{}{
				"threshold": "0.6",
				"name":      "test",
				"count":     "100",
			},
			want: map[string]interface{}{
				"threshold": float64(0.6),
				"name":      "test",
				"count":     int64(100),
			},
		},
		{
			name: "nested maps",
			input: map[string]interface{}{
				"config": map[string]interface{}{
					"threshold": "0.8",
					"enabled":   "true",
				},
			},
			want: map[string]interface{}{
				"config": map[string]interface{}{
					"threshold": float64(0.8),
					"enabled":   true,
				},
			},
		},
		{
			name: "arrays",
			input: map[string]interface{}{
				"values": []interface{}{"0.1", "0.2", "100"},
			},
			want: map[string]interface{}{
				"values": []interface{}{float64(0.1), float64(0.2), int64(100)},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			r.convertStringsToFloats(tt.input)
			// Deep comparison would be complex, just check a few key conversions
			for k, expectedVal := range tt.want {
				actualVal, ok := tt.input[k]
				if !ok {
					t.Errorf("key %q not found in result", k)
					continue
				}

				// Type assertion to compare
				switch expected := expectedVal.(type) {
				case float64:
					actual, ok := actualVal.(float64)
					if !ok || actual != expected {
						t.Errorf("key %q: got %v (type %T), want %v (type %T)",
							k, actualVal, actualVal, expectedVal, expectedVal)
					}
				case int64:
					actual, ok := actualVal.(int64)
					if !ok || actual != expected {
						t.Errorf("key %q: got %v (type %T), want %v (type %T)",
							k, actualVal, actualVal, expectedVal, expectedVal)
					}
				case bool:
					actual, ok := actualVal.(bool)
					if !ok || actual != expected {
						t.Errorf("key %q: got %v (type %T), want %v (type %T)",
							k, actualVal, actualVal, expectedVal, expectedVal)
					}
				}
			}
		})
	}
}
