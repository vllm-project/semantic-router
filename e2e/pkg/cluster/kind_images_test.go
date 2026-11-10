package cluster

import (
	"context"
	"slices"
	"testing"
)

func TestTagLoadedImageTagsInsideTheNodesContainerdNamespace(t *testing.T) {
	got := tagLoadedImageArgs("e2e-control-plane", "repo/router:e2e-test", "repo/router:latest")
	want := []string{
		"exec", "e2e-control-plane", "ctr", "--namespace", "k8s.io",
		"images", "tag", "--force", "repo/router:e2e-test", "repo/router:latest",
	}
	if !slices.Equal(got, want) {
		t.Fatalf("tagLoadedImageArgs() = %v, want %v", got, want)
	}
}

func TestTagLoadedImageSkipsAnIdenticalName(t *testing.T) {
	if err := TagLoadedImage(context.Background(), "no-such-cluster", "repo/router:latest", "repo/router:latest"); err != nil {
		t.Fatalf("an identical name needs no tag: %v", err)
	}
}
