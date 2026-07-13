package proxy

import (
	"strings"
	"testing"
)

func TestProxyConstructorsRejectCredentialedTargetsWithoutEchoingCredentials(t *testing.T) {
	constructors := map[string]func(string) error{
		"HTTP": func(target string) error {
			_, err := NewReverseProxy(target, "/embedded/test", false)
			return err
		},
		"Jaeger": func(target string) error {
			_, err := NewJaegerProxy(target, "/embedded/jaeger")
			return err
		},
		"WebSocket": func(target string) error {
			_, err := NewWebSocketAwareHandler(target, "/embedded/test")
			return err
		},
	}
	targets := []string{
		"http://proxy-user:proxy-secret@upstream.example/base?token=target-query-secret",
		"http://proxy-user:proxy%zz@upstream.example/base",
	}

	for name, construct := range constructors {
		for _, target := range targets {
			t.Run(name+"/"+target, func(t *testing.T) {
				err := construct(target)
				if err == nil {
					t.Fatalf("constructor accepted credentialed target %q", target)
				}
				for _, secret := range []string{
					"proxy-user",
					"proxy-secret",
					"target-query-secret",
				} {
					if strings.Contains(err.Error(), secret) {
						t.Fatalf("constructor error leaked %q: %v", secret, err)
					}
				}
			})
		}
	}
}
