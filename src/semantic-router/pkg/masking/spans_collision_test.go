package masking

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestMaskText_IntersectingTemplatesKeepValuesDistinct(t *testing.T) {
	for _, tc := range []struct {
		name, first, second, reserved, wantFirst, wantSecond string
	}{
		{"forward", "first", "second", "", "[V0X0]", "[V0X1]"},
		{"reverse", "second", "first", "", "[V0X0]", "[V1X0]"},
		{"issued_and_reserved", "first", "second", "[V0X1]", "[V0X0]", "[V0X2]"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			a := NewAllocator(&config.MaskingPluginConfig{
				Rules: []config.SignalRef{{Type: config.SignalTypeKeyword, Name: "first"}, {Type: config.SignalTypeKeyword, Name: "second"}},
				Placeholders: map[string]string{
					"first":  "[V{index}X0]",
					"second": "[V0X{index}]",
				},
			})
			if tc.reserved != "" {
				if _, _, err := MaskText(tc.reserved, nil, nil, a); err != nil {
					t.Fatal(err)
				}
			}
			// One allocator spans all messages/dispatches in the request.
			for _, call := range []struct {
				text, want string
				spans      []Span
			}{
				{"alpha", tc.wantFirst, []Span{{EntityType: tc.first, Start: 0, End: 5}}},
				{"beta", tc.wantSecond, []Span{{EntityType: tc.second, Start: 0, End: 4}}},
				{"alpha beta", tc.wantFirst + " " + tc.wantSecond, []Span{
					{EntityType: tc.first, Start: 0, End: 5},
					{EntityType: tc.second, Start: 6, End: 10},
				}},
			} {
				got, _, err := MaskText(call.text, call.spans, nil, a)
				if err != nil {
					t.Fatal(err)
				}
				if got != call.want {
					t.Fatalf("MaskText(%q) = %q, want %q", call.text, got, call.want)
				}
			}
		})
	}
}
