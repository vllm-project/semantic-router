package main

import (
	"flag"
	"testing"
)

func TestBoolFlagOverrideOnlyWhenExplicitlySet(t *testing.T) {
	fs := flag.NewFlagSet("test", flag.ContinueOnError)
	value := fs.Bool("management-remote-exposure", false, "")
	if err := fs.Parse([]string{}); err != nil {
		t.Fatalf("Parse() error = %v", err)
	}
	if got := boolFlagOverride(fs, "management-remote-exposure", *value); got != nil {
		t.Fatalf("unset flag override = %v, want nil", *got)
	}

	fs = flag.NewFlagSet("test", flag.ContinueOnError)
	value = fs.Bool("management-remote-exposure", false, "")
	if err := fs.Parse([]string{"-management-remote-exposure=true"}); err != nil {
		t.Fatalf("Parse() error = %v", err)
	}
	got := boolFlagOverride(fs, "management-remote-exposure", *value)
	if got == nil || !*got {
		t.Fatalf("explicit true override = %v, want true", got)
	}

	fs = flag.NewFlagSet("test", flag.ContinueOnError)
	value = fs.Bool("management-remote-exposure", false, "")
	if err := fs.Parse([]string{"-management-remote-exposure=false"}); err != nil {
		t.Fatalf("Parse() error = %v", err)
	}
	got = boolFlagOverride(fs, "management-remote-exposure", *value)
	if got == nil || *got {
		t.Fatalf("explicit false override = %v, want false", got)
	}
}
