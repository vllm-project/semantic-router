package principal

import (
	"context"
	"errors"
	"testing"
)

func TestDelegatedFollowsTheCallsPrincipal(t *testing.T) {
	background := context.Background()
	if _, err := Delegated(background); !errors.Is(err, ErrRouterCall) {
		t.Fatalf("a call with no principal acts as the Router: got %v", err)
	}

	user := ForRequest(background, "alice-key")
	if credential, err := Delegated(user); err != nil || credential != "alice-key" {
		t.Fatalf("a user call presents the request's credential: got %q, %v", credential, err)
	}
	if Refused(user) {
		t.Fatal("a call that presented its credential was not refused")
	}

	detached := Inherit(context.WithoutCancel(background), user)
	if credential, _ := Delegated(detached); credential != "alice-key" {
		t.Fatalf("a detached call made for the request keeps its principal: got %q", credential)
	}
	if _, err := Delegated(ForRouter(user)); !errors.Is(err, ErrRouterCall) {
		t.Fatalf("a Router-own call derived from a request acts as the Router: got %v", err)
	}

	missing := ForRequest(background, "")
	if _, err := Delegated(missing); !errors.Is(err, ErrCredentialMissing) {
		t.Fatalf("a user call without a credential is refused for it: got %v", err)
	}
	if !Refused(missing) || !ForUser(missing) || ForUser(background) {
		t.Fatal("the request records the refusal, and only a request's calls act for a user")
	}
}

func TestRequestHeaderFollowsTheCallsPrincipal(t *testing.T) {
	background := context.Background()
	user := ForRequestWithHeaders(background, "", map[string]string{"x-user-id": "alice"})
	if value, ok := RequestHeader(user, " X-User-Id "); !ok || value != "alice" {
		t.Fatalf("a user call copies the request's header in any letter case: got %q, %v", value, ok)
	}
	if value, _ := RequestHeader(Inherit(context.WithoutCancel(background), user), "x-user-id"); value != "alice" {
		t.Fatalf("a detached call made for the request keeps its headers: got %q", value)
	}
	for name, ctx := range map[string]context.Context{
		"a Router-own call":                     background,
		"a Router-own call derived from it":     ForRouter(user),
		"a request that carried no such header": ForRequest(background, "alice-key"),
	} {
		if value, ok := RequestHeader(ctx, "x-user-id"); ok {
			t.Fatalf("%s has no request header to copy: got %q", name, value)
		}
	}
	if _, ok := RequestHeader(user, "x-other"); ok {
		t.Fatal("only the headers the principal carries are copied")
	}
}
