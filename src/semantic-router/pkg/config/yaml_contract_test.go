package config

import (
	"strings"
	"testing"
)

type yamlBooleanFixture struct {
	Enabled bool `yaml:"enabled"`
}

func TestDecodeYAML12StrictPreservesConditionKeys(t *testing.T) {
	var decoded struct {
		Retry struct {
			On []string `yaml:"on"`
		} `yaml:"retry"`
	}
	if err := DecodeYAML12Strict([]byte("retry:\n  on: [unavailable, timeout]\n"), &decoded); err != nil {
		t.Fatal(err)
	}
	if got := strings.Join(decoded.Retry.On, ","); got != "unavailable,timeout" {
		t.Fatalf("retry.on = %q", got)
	}
}

func TestDecodeYAML12StrictRejectsYAML11BooleanSpelling(t *testing.T) {
	var decoded yamlBooleanFixture
	if err := DecodeYAML12Strict([]byte("enabled: yes\n"), &decoded); err == nil {
		t.Fatal("DecodeYAML12Strict accepted the YAML 1.1 boolean spelling yes")
	}
}

func TestDecodeYAML12StrictRejectsDuplicateKeys(t *testing.T) {
	var decoded yamlBooleanFixture
	err := DecodeYAML12Strict([]byte("enabled: true\nenabled: false\n"), &decoded)
	if err == nil || !strings.Contains(err.Error(), "mapping key \"enabled\" already defined") {
		t.Fatalf("DecodeYAML12Strict() error = %v", err)
	}
}

func TestFullConfigRejectsDuplicateKeys(t *testing.T) {
	document := strings.Replace(
		strictV03AuthoringYAML,
		"version: v0.3",
		"version: v0.3\nversion: v0.3",
		1,
	)
	_, err := testAuthoringParser(t).ParseYAMLBytes([]byte(document))
	if err == nil || !strings.Contains(err.Error(), "mapping key \"version\" already defined") {
		t.Fatalf("ParseYAMLBytes() error = %v", err)
	}
}

func TestRoutingFragmentRejectsDuplicateKeys(t *testing.T) {
	_, err := ParseRoutingYAMLBytes([]byte("routing:\n  signals: {}\n  signals: {}\n"))
	if err == nil || !strings.Contains(err.Error(), "mapping key \"signals\" already defined") {
		t.Fatalf("ParseRoutingYAMLBytes() error = %v", err)
	}
}
