package config

import (
	"reflect"
	"testing"
)

func TestIdentityHeaderNames(t *testing.T) {
	builtIn := []string{"x-authz-user-id", "x-authz-user-groups", "x-authz-team-id", "x-authz-tenant-id"}
	if got := (IdentityConfig{}).HeaderNames(); !reflect.DeepEqual(got, builtIn) {
		t.Fatalf("default HeaderNames() = %v, want %v", got, builtIn)
	}
	got := IdentityConfig{UserIDHeader: " X-JWT-Sub ", UserGroupsHeader: "x-authz-user-groups"}.HeaderNames()
	if want := append(append([]string(nil), builtIn...), "x-jwt-sub"); !reflect.DeepEqual(got, want) {
		t.Fatalf("configured HeaderNames() = %v, want %v", got, want)
	}
}
