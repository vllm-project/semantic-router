package testcases

import (
	"context"
	"errors"
	"fmt"
	"time"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"

	"github.com/vllm-project/semantic-router/e2e/pkg/helm"
	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

// The upgrade case runs its own release beside the profile's. Its name differs
// from the profile's so the chart's cluster-scoped RBAC does not collide.
const (
	upgradeRelease        = "semantic-router-upgrade"
	upgradeNamespace      = "vllm-sr-helm-upgrade"
	publishedChart        = "oci://ghcr.io/vllm-project/charts/semantic-router"
	publishedChartVersion = "0.0.0-latest"
	standaloneValuesFile  = "e2e/profiles/standalone/values.yaml"
)

func init() {
	pkgtestcases.Register("standalone-helm-upgrade", pkgtestcases.TestCase{
		Description:         "helm upgrade from the chart main last published to the source chart rolls the Router out and keeps routing",
		Tags:                []string{"standalone", "helm", "kubernetes"},
		MutatesClusterState: true,
		Fn:                  testStandaloneHelmUpgrade,
	})
}

// testStandaloneHelmUpgrade installs the published chart with the profile's
// values, upgrades that release in place to the source chart, and requires
// the Router it rolls out to route a chat completion. Both charts default to
// the development image, which the profile points at this run's Router.
func testStandaloneHelmUpgrade(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) (err error) {
	deployer := helm.NewDeployer(opts.KubeConfig, opts.Verbose)
	// The release owns cluster-scoped RBAC that deleting the namespace cannot
	// remove, so a failed uninstall fails the case instead of leaking it.
	defer func() {
		cleanupCtx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
		defer cancel()
		if uninstallErr := deployer.Uninstall(cleanupCtx, upgradeRelease, upgradeNamespace); uninstallErr != nil {
			err = errors.Join(err, fmt.Errorf("uninstall the upgrade release: %w", uninstallErr))
		}
		deleteErr := client.CoreV1().Namespaces().Delete(cleanupCtx, upgradeNamespace, metav1.DeleteOptions{})
		if deleteErr != nil && !apierrors.IsNotFound(deleteErr) {
			err = errors.Join(err, fmt.Errorf("delete the upgrade namespace: %w", deleteErr))
		}
	}()

	release := helm.InstallOptions{
		ReleaseName: upgradeRelease,
		Chart:       publishedChart,
		Version:     publishedChartVersion,
		Namespace:   upgradeNamespace,
		ValuesFiles: []string{standaloneValuesFile},
	}
	if err = deployer.Install(ctx, release); err != nil {
		return fmt.Errorf("install the published chart: %w", err)
	}
	if err = waitForRollout(ctx, client, upgradeNamespace, upgradeRelease, 30*time.Minute); err != nil {
		return fmt.Errorf("wait for the published chart's Router: %w", err)
	}

	release.Chart = helm.SemanticRouterRelease.Chart
	release.Version = ""
	if err = deployer.Install(ctx, release); err != nil {
		return fmt.Errorf("upgrade to the source chart: %w", err)
	}
	if err = waitForRollout(ctx, client, upgradeNamespace, upgradeRelease, 30*time.Minute); err != nil {
		return fmt.Errorf("wait for the upgraded Router: %w", err)
	}

	upgraded := opts
	upgraded.ServiceConfig.Name = upgradeRelease
	upgraded.ServiceConfig.Namespace = upgradeNamespace
	reply, err := standaloneChat(ctx, client, upgraded, "Which model answers after the upgrade?")
	if err != nil {
		return fmt.Errorf("after the upgrade: %w", err)
	}
	return reply.servedBy(standalonePrimaryModel, standaloneDefaultRoute)
}
