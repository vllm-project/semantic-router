package cluster

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
)

// EnsureSchedulableWorkloadNodes makes sure generic workloads (e.g. Helm releases for
// semantic-router) can be scheduled on this Kind cluster.
//
// Kind normally adds a worker node without the control-plane NoSchedule taint; however
// some environments (Podman provider, partial cluster state, or very old clusters)
// end up with only a Ready control-plane node that still has the default taint. In
// that case Helm --wait will hang while pods stay Pending.
//
// If at least one Ready node exists without control-plane/master NoSchedule taints,
// this is a no-op. Otherwise it removes those NoSchedule taints from Ready nodes.
// This is intended only for local E2E/CI Kind clusters, not production.
func (k *KindCluster) EnsureSchedulableWorkloadNodes(ctx context.Context) error {
	kubeConfig, err := k.GetKubeConfig(ctx)
	if err != nil {
		return fmt.Errorf("get kubeconfig for schedulability check: %w", err)
	}
	defer func() { _ = os.Remove(kubeConfig) }()

	cmd := exec.CommandContext(ctx, "kubectl", "--kubeconfig", kubeConfig, "get", "nodes", "-o", "json")
	out, err := cmd.Output()
	if err != nil {
		return fmt.Errorf("kubectl get nodes: %w", err)
	}

	var nl nodeListJSON
	if err := json.Unmarshal(out, &nl); err != nil {
		return fmt.Errorf("parse kubectl nodes json: %w", err)
	}

	if hasReadyNodeWithoutControlPlaneNoSchedule(&nl) {
		k.log("Schedulability: found Ready node(s) without control-plane NoSchedule taint; no change")
		return nil
	}

	k.log("Schedulability: no Ready non-control-plane node; removing control-plane NoSchedule taint(s) for E2E Kind compatibility")
	for i := range nl.Items {
		node := &nl.Items[i]
		if !node.isReady() {
			continue
		}
		name := node.Metadata.Name
		if name == "" {
			continue
		}
		for _, key := range []string{
			"node-role.kubernetes.io/control-plane",
			"node-role.kubernetes.io/master",
		} {
			taintArg := key + ":NoSchedule-"
			c := exec.CommandContext(ctx, "kubectl", "--kubeconfig", kubeConfig,
				"taint", "nodes", name, taintArg)
			if k.Verbose {
				c.Stderr = os.Stderr
				c.Stdout = os.Stdout
			}
			_ = c.Run()
		}
	}

	return nil
}

type nodeListJSON struct {
	Items []nodeItemJSON `json:"items"`
}

type nodeItemJSON struct {
	Metadata struct {
		Name string `json:"name"`
	} `json:"metadata"`
	Spec struct {
		Taints []struct {
			Key    string `json:"key"`
			Effect string `json:"effect"`
		} `json:"taints"`
	} `json:"spec"`
	Status struct {
		Conditions []struct {
			Type   string `json:"type"`
			Status string `json:"status"`
		} `json:"conditions"`
	} `json:"status"`
}

func (n *nodeItemJSON) isReady() bool {
	for _, c := range n.Status.Conditions {
		if c.Type == "Ready" && c.Status == "True" {
			return true
		}
	}
	return false
}

func hasControlPlaneNoScheduleTaint(node *nodeItemJSON) bool {
	for _, t := range node.Spec.Taints {
		if t.Effect != "NoSchedule" {
			continue
		}
		if t.Key == "node-role.kubernetes.io/control-plane" || t.Key == "node-role.kubernetes.io/master" {
			return true
		}
	}
	return false
}

func hasReadyNodeWithoutControlPlaneNoSchedule(nl *nodeListJSON) bool {
	for i := range nl.Items {
		node := &nl.Items[i]
		if !node.isReady() {
			continue
		}
		if !hasControlPlaneNoScheduleTaint(node) {
			return true
		}
	}
	return false
}
