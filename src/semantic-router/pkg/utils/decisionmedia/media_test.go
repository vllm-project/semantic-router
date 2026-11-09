package decisionmedia

import (
	"encoding/base64"
	"reflect"
	"strings"
	"testing"
)

func png(s string) *Source {
	return &Source{Kind: "image", MediaType: "image/png", Data: base64.StdEncoding.EncodeToString([]byte("\x89PNG\r\n\x1a\n" + s))}
}

func TestOrderAndSkips(t *testing.T) {
	a, b, c := png("a"), png("b"), png("c")
	counts := map[string]int{}
	sources := []*Source{a, b, {Kind: "image", URL: "https://example.invalid/x"}, {Kind: "image", MediaType: "image/gif", Data: "R0lGODlh"}, c, {Kind: "image", MediaType: "image/png", Data: "invalid"}}
	got := Select(sources, Policy{Images: &Limit{Count: 2, Bytes: 100}}, func(_, r string) { counts[r]++ })
	if !reflect.DeepEqual(got.Images, []string{b.value, c.value}) {
		t.Fatal("order")
	}
	for _, r := range []string{"malformed", "remote_url", "unsupported_type", "over_count"} {
		if counts[r] != 1 {
			t.Fatal(counts)
		}
	}
	if a.value != "" {
		t.Fatal("decoded excess attachment")
	}
}

func TestLimitsAndIsolation(t *testing.T) {
	huge := &Source{Kind: "image", MediaType: "image/png", Data: strings.Repeat("A", base64.StdEncoding.EncodedLen(ImageMaxBytes)+1)}
	if got := Select([]*Source{huge}, Policy{Images: &Limit{Count: 4, Bytes: ImageMaxBytes}}, nil); len(got.Images) != 0 || huge.reason != "too_large" {
		t.Fatal("encoded cap")
	}
	a, b := png("old"), png("new")
	b.once.Do(b.validate)
	got := Select([]*Source{a, b}, Policy{Images: &Limit{Count: 4, Bytes: 100}, EncodedBytes: len(b.value) + 3}, nil)
	if !reflect.DeepEqual(got.Images, []string{b.value}) {
		t.Fatal("aggregate cap")
	}
	if got := Select([]*Source{a}, Policy{Images: &Limit{Count: 4, Bytes: 8}}, nil); len(got.Images) != 0 {
		t.Fatal("lower cap")
	}
	if got := Select([]*Source{a}, Policy{Images: &Limit{Count: 4, Bytes: 100}}, nil); len(got.Images) != 1 {
		t.Fatal("deployment isolation")
	}
	if got := Select([]*Source{a}, Policy{}, nil); len(got.Images) != 0 {
		t.Fatal("default off")
	}
	fake := &Source{Kind: "image", MediaType: "image/png", Data: "bm90IHBuZw=="}
	Select([]*Source{fake}, Policy{Images: &Limit{Count: 4, Bytes: 100}}, nil)
	if fake.reason != "malformed" {
		t.Fatal("signature not checked")
	}
}
