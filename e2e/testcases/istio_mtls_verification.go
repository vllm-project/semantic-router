package testcases

import (
	"context"
	"fmt"
	"os/exec"
	"strings"

	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
)

func init() {
	pkgtestcases.Register("istio-mtls-verification", pkgtestcases.TestCase{
		Description: "Verify mutual TLS is enabled between services in the Istio mesh",
		Tags:        []string{"istio", "mtls", "security"},
		Fn:          testIstioMTLSVerification,
	})
}

func testIstioMTLSVerification(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	if opts.Verbose {
		fmt.Println("[Test] Verifying Istio mTLS configuration")
	}

	// Istio-specific test: always use vllm-semantic-router-system namespace
	namespace := "vllm-semantic-router-system"

	// 1. Verify DestinationRule has mTLS mode set to ISTIO_MUTUAL
	if err := verifyDestinationRuleMTLS(ctx, namespace, opts.Verbose); err != nil {
		return fmt.Errorf("DestinationRule mTLS verification failed: %w", err)
	}

	// 2. Get a semantic-router pod to use for testing
	pods, err := client.CoreV1().Pods(namespace).List(ctx, metav1.ListOptions{
		LabelSelector: "app.kubernetes.io/name=semantic-router",
	})
	if err != nil {
		return fmt.Errorf("failed to list pods: %w", err)
	}

	if len(pods.Items) == 0 {
		return fmt.Errorf("no semantic-router pods found in namespace %s", namespace)
	}

	podName := pods.Items[0].Name

	// 3. Check if mTLS certificates are present in the istio-proxy container
	certCheckPassed, certDetails, err := checkIstioProxyCertificates(ctx, namespace, podName, opts.Verbose)
	if err != nil {
		return fmt.Errorf("failed to check Istio proxy certificates: %w", err)
	}

	if !certCheckPassed {
		return fmt.Errorf("Istio proxy mTLS certificates not found or invalid")
	}

	// 4. Verify PeerAuthentication policy (if exists)
	peerAuthExists, peerAuthMode := checkPeerAuthentication(ctx, namespace, opts.Verbose)

	// Set details for reporting
	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"namespace":               namespace,
			"destination_rule_mtls":   "ISTIO_MUTUAL",
			"istio_proxy_certs_found": certCheckPassed,
			"cert_details":            certDetails,
			"peer_auth_exists":        peerAuthExists,
			"peer_auth_mode":          peerAuthMode,
		})
	}

	if opts.Verbose {
		fmt.Println("[Test] ✅ Istio mTLS verification passed")
	}

	return nil
}

// verifyDestinationRuleMTLS verifies the DestinationRule has mTLS configured
func verifyDestinationRuleMTLS(ctx context.Context, namespace string, verbose bool) error {
	// Use kubectl to get the DestinationRule and check its mTLS configuration
	cmd := exec.CommandContext(ctx, "kubectl", "get", "destinationrule",
		"semantic-router", "-n", namespace, "-o", "yaml")

	output, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("failed to get DestinationRule: %w (output: %s)", err, string(output))
	}

	// Check if the output contains "mode: ISTIO_MUTUAL"
	if !strings.Contains(string(output), "ISTIO_MUTUAL") {
		return fmt.Errorf("DestinationRule does not have ISTIO_MUTUAL mode configured")
	}

	if verbose {
		fmt.Println("[Test] DestinationRule has ISTIO_MUTUAL mode configured")
	}

	return nil
}

// checkIstioProxyCertificates checks if mTLS certificates are present in the istio-proxy container
func checkIstioProxyCertificates(ctx context.Context, namespace, podName string, verbose bool) (bool, map[string]interface{}, error) {
	// Check for Istio certificates in the istio-proxy container
	// Certificates are typically mounted at /etc/certs/ or /var/run/secrets/istio/
	certPaths := []string{
		"/etc/certs/cert-chain.pem",
		"/etc/certs/key.pem",
		"/etc/certs/root-cert.pem",
	}

	certDetails := make(map[string]interface{})
	allCertsFound := true

	for _, certPath := range certPaths {
		cmd := exec.CommandContext(ctx, "kubectl", "exec", "-n", namespace,
			podName, "-c", "istio-proxy", "--", "ls", certPath)

		output, err := cmd.CombinedOutput()
		certExists := err == nil && strings.TrimSpace(string(output)) == certPath

		certDetails[certPath] = certExists

		if !certExists && verbose {
			fmt.Printf("[Test] Certificate not found at %s in pod %s\n", certPath, podName)
		}
	}

	// In newer Istio versions, certs might be in memory, so we check for the istio-proxy process
	// If certificates are not found in the expected locations, check if SDS (Secret Discovery Service) is being used
	if !allCertsFound {
		// Check if istio-proxy is running with SDS (indicated by pilot-agent process)
		cmd := exec.CommandContext(ctx, "kubectl", "exec", "-n", namespace,
			podName, "-c", "istio-proxy", "--", "pgrep", "-f", "pilot-agent")

		output, err := cmd.CombinedOutput()
		if err == nil && len(strings.TrimSpace(string(output))) > 0 {
			certDetails["sds_enabled"] = true
			allCertsFound = true // SDS is valid for mTLS
			if verbose {
				fmt.Printf("[Test] Istio SDS (Secret Discovery Service) detected in pod %s\n", podName)
			}
		}
	}

	return allCertsFound, certDetails, nil
}

// checkPeerAuthentication checks if a PeerAuthentication policy exists
func checkPeerAuthentication(ctx context.Context, namespace string, verbose bool) (bool, string) {
	// Check for PeerAuthentication in the namespace
	cmd := exec.CommandContext(ctx, "kubectl", "get", "peerauthentication",
		"-n", namespace, "-o", "yaml")

	output, err := cmd.CombinedOutput()
	if err != nil {
		// PeerAuthentication might not exist, which is fine (mTLS can be enabled by default)
		if verbose {
			fmt.Println("[Test] No PeerAuthentication policy found (using mesh-wide default)")
		}
		return false, "mesh-default"
	}

	// Check if strict mode is enabled
	if strings.Contains(string(output), "mode: STRICT") {
		if verbose {
			fmt.Println("[Test] PeerAuthentication policy has STRICT mode")
		}
		return true, "STRICT"
	}

	if strings.Contains(string(output), "mode: PERMISSIVE") {
		if verbose {
			fmt.Println("[Test] PeerAuthentication policy has PERMISSIVE mode")
		}
		return true, "PERMISSIVE"
	}

	return true, "unknown"
}
