package main

import (
	"errors"
	"strings"
	"testing"
)

func TestResolveDSNUsesExactlyOneSecretReference(t *testing.T) {
	lookup := func(name string) (string, bool) {
		if name == "ROUTER_POSTGRES_DSN" {
			return "postgres://router:secret@db/router?sslmode=require", true
		}
		return "", false
	}
	read := func(path string) ([]byte, error) {
		if path != "/run/secrets/router-postgres-dsn" {
			return nil, errors.New("missing")
		}
		return []byte("postgres://router:secret@db/router\n"), nil
	}

	fromEnv, err := resolveDSN(options{dsnEnv: "ROUTER_POSTGRES_DSN"}, lookup, read)
	if err != nil || !strings.HasPrefix(fromEnv, "postgres://") {
		t.Fatalf("resolveDSN(env) = (%q, %v)", fromEnv, err)
	}
	fromFile, err := resolveDSN(options{dsnFile: "/run/secrets/router-postgres-dsn"}, lookup, read)
	if err != nil || fromFile != "postgres://router:secret@db/router" {
		t.Fatalf("resolveDSN(file) = (%q, %v)", fromFile, err)
	}
	for _, opts := range []options{
		{},
		{dsnFile: "/run/secrets/dsn", dsnEnv: "DSN"},
	} {
		if _, err := resolveDSN(opts, lookup, read); err == nil {
			t.Fatalf("resolveDSN(%+v) unexpectedly succeeded", opts)
		}
	}
}

func TestResolveDSNNeverReturnsRawReadErrors(t *testing.T) {
	secretError := errors.New("secret backend revealed sensitive/path/details")
	_, err := resolveDSN(options{dsnFile: "/run/secrets/dsn"}, func(string) (string, bool) { return "", false }, func(string) ([]byte, error) {
		return nil, secretError
	})
	if err == nil || strings.Contains(err.Error(), secretError.Error()) {
		t.Fatalf("resolveDSN() error = %v", err)
	}
}
