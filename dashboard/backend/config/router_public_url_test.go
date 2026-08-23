package config

import "testing"

func TestCanonicalRouterPublicURL(t *testing.T) {
	t.Parallel()

	for _, testCase := range []struct {
		name  string
		value string
		want  string
		ok    bool
	}{
		{name: "same origin ingress", value: "", want: "", ok: true},
		{name: "https origin", value: "https://Router.Example.test/", want: "https://router.example.test", ok: true},
		{name: "local development", value: "http://localhost:8080", want: "http://localhost:8080", ok: true},
		{name: "credentials", value: "https://user:secret@router.example.test", ok: false},
		{name: "path", value: "https://router.example.test/gateway", ok: false},
		{name: "query", value: "https://router.example.test?token=value", ok: false},
	} {
		t.Run(testCase.name, func(t *testing.T) {
			t.Parallel()
			got, err := canonicalRouterPublicURL(testCase.value)
			if (err == nil) != testCase.ok {
				t.Fatalf("canonicalRouterPublicURL(%q) error=%v", testCase.value, err)
			}
			if err == nil && got != testCase.want {
				t.Fatalf("canonicalRouterPublicURL(%q)=%q, want %q", testCase.value, got, testCase.want)
			}
		})
	}
}
