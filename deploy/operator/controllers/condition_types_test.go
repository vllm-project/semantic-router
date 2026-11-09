package controllers

import (
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"strings"
	"testing"

	routerconfig "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// A decision condition of the CRD names a type the Router knows: every
// signal type of its catalog, and projection.
func TestCRDConditionTypesAreTheRouterSignalTypes(t *testing.T) {
	source, err := os.ReadFile(filepath.Join("..", "api", "v1alpha1", "semanticrouter_types.go"))
	if err != nil {
		t.Fatal(err)
	}
	match := regexp.MustCompile(`(?s)type RuleConditionConfig struct \{.*?\+kubebuilder:validation:Enum=([a-z_;]+)`).FindSubmatch(source)
	if match == nil {
		t.Fatal("RuleConditionConfig declares no condition type enum")
	}
	got := strings.Split(string(match[1]), ";")
	want := append(routerconfig.SupportedSignalTypes(), "projection")
	slices.Sort(got)
	slices.Sort(want)
	if !slices.Equal(got, want) {
		t.Fatalf("condition types = %v, want the Router's %v", got, want)
	}
}
