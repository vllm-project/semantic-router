package cluster

import (
	"context"
	"fmt"
	"os/exec"
	"strings"
)

// TagLoadedImage gives an image that is already loaded into a Kind cluster a
// second name on every node, so a chart default resolves to it without
// changing any tag on the host.
func TagLoadedImage(ctx context.Context, clusterName, source, target string) error {
	if source == target {
		return nil
	}
	output, err := exec.CommandContext(ctx, "kind", "get", "nodes", "--name", clusterName).Output() //nolint:gosec // The cluster name comes from the E2E run, never from a request.
	if err != nil {
		return fmt.Errorf("list the nodes of Kind cluster %s: %w", clusterName, err)
	}
	nodes := strings.Fields(string(output))
	if len(nodes) == 0 {
		return fmt.Errorf("kind cluster %s has no nodes", clusterName)
	}
	for _, node := range nodes {
		tag := exec.CommandContext(ctx, "docker", tagLoadedImageArgs(node, source, target)...) //nolint:gosec // Node and image names come from the E2E run.
		if output, err := tag.CombinedOutput(); err != nil {
			return fmt.Errorf("tag %s as %s on node %s: %w: %s", source, target, node, err, strings.TrimSpace(string(output)))
		}
	}
	return nil
}

func tagLoadedImageArgs(node, source, target string) []string {
	return []string{"exec", node, "ctr", "--namespace", "k8s.io", "images", "tag", "--force", source, target}
}
