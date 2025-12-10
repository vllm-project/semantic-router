package performance

import (
	"context"
	"fmt"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
	metricsv "k8s.io/metrics/pkg/client/clientset/versioned"
)

// MetricsCollector collects performance metrics from Kubernetes pods
type MetricsCollector struct {
	kubeClient    *kubernetes.Clientset
	metricsClient *metricsv.Clientset
	namespace     string
}

// NewMetricsCollector creates a new metrics collector
func NewMetricsCollector(kubeClient *kubernetes.Clientset, metricsClient *metricsv.Clientset, namespace string) *MetricsCollector {
	return &MetricsCollector{
		kubeClient:    kubeClient,
		metricsClient: metricsClient,
		namespace:     namespace,
	}
}

// PodMetrics holds metrics for a single pod
type PodMetrics struct {
	PodName        string
	Timestamp      time.Time
	CPUUsageCores  float64
	MemoryUsageMB  float64
	ContainerCount int
}

// CollectPodMetrics collects metrics for a specific pod
func (mc *MetricsCollector) CollectPodMetrics(ctx context.Context, podName string) (*PodMetrics, error) {
	if mc.metricsClient == nil {
		return nil, fmt.Errorf("metrics client not available")
	}

	podMetrics, err := mc.metricsClient.MetricsV1beta1().PodMetricses(mc.namespace).Get(ctx, podName, metav1.GetOptions{})
	if err != nil {
		return nil, fmt.Errorf("failed to get pod metrics: %w", err)
	}

	metrics := &PodMetrics{
		PodName:        podName,
		Timestamp:      podMetrics.Timestamp.Time,
		ContainerCount: len(podMetrics.Containers),
	}

	// Aggregate CPU and memory across all containers
	for _, container := range podMetrics.Containers {
		cpuQuantity := container.Usage.Cpu()
		memQuantity := container.Usage.Memory()

		// Convert to float64
		metrics.CPUUsageCores += float64(cpuQuantity.MilliValue()) / 1000.0
		metrics.MemoryUsageMB += float64(memQuantity.Value()) / (1024 * 1024)
	}

	return metrics, nil
}

// CollectPodMetricsByLabel collects metrics for all pods matching a label selector
func (mc *MetricsCollector) CollectPodMetricsByLabel(ctx context.Context, labelSelector string) ([]*PodMetrics, error) {
	pods, err := mc.kubeClient.CoreV1().Pods(mc.namespace).List(ctx, metav1.ListOptions{
		LabelSelector: labelSelector,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to list pods: %w", err)
	}

	var allMetrics []*PodMetrics
	for _, pod := range pods.Items {
		metrics, err := mc.CollectPodMetrics(ctx, pod.Name)
		if err != nil {
			// Log error but continue with other pods
			fmt.Printf("Warning: failed to collect metrics for pod %s: %v\n", pod.Name, err)
			continue
		}
		allMetrics = append(allMetrics, metrics)
	}

	return allMetrics, nil
}

// MonitorPodMetrics continuously monitors pod metrics during a test
func (mc *MetricsCollector) MonitorPodMetrics(ctx context.Context, podName string, interval time.Duration, results chan<- *PodMetrics) {
	ticker := time.NewTicker(interval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			metrics, err := mc.CollectPodMetrics(ctx, podName)
			if err != nil {
				fmt.Printf("Warning: failed to collect metrics: %v\n", err)
				continue
			}
			results <- metrics
		}
	}
}

// ResourceStats holds aggregated resource statistics
type ResourceStats struct {
	AvgCPUCores float64
	MaxCPUCores float64
	MinCPUCores float64
	AvgMemoryMB float64
	MaxMemoryMB float64
	MinMemoryMB float64
	SampleCount int
}

// AggregateMetrics aggregates multiple pod metrics samples
func AggregateMetrics(metrics []*PodMetrics) *ResourceStats {
	if len(metrics) == 0 {
		return &ResourceStats{}
	}

	stats := &ResourceStats{
		MinCPUCores: metrics[0].CPUUsageCores,
		MaxCPUCores: metrics[0].CPUUsageCores,
		MinMemoryMB: metrics[0].MemoryUsageMB,
		MaxMemoryMB: metrics[0].MemoryUsageMB,
		SampleCount: len(metrics),
	}

	var totalCPU, totalMem float64

	for _, m := range metrics {
		totalCPU += m.CPUUsageCores
		totalMem += m.MemoryUsageMB

		if m.CPUUsageCores < stats.MinCPUCores {
			stats.MinCPUCores = m.CPUUsageCores
		}
		if m.CPUUsageCores > stats.MaxCPUCores {
			stats.MaxCPUCores = m.CPUUsageCores
		}

		if m.MemoryUsageMB < stats.MinMemoryMB {
			stats.MinMemoryMB = m.MemoryUsageMB
		}
		if m.MemoryUsageMB > stats.MaxMemoryMB {
			stats.MaxMemoryMB = m.MemoryUsageMB
		}
	}

	stats.AvgCPUCores = totalCPU / float64(len(metrics))
	stats.AvgMemoryMB = totalMem / float64(len(metrics))

	return stats
}

// PrintResourceStats prints resource statistics
func (rs *ResourceStats) PrintResourceStats() {
	fmt.Println("\n" + "===================================================================================")
	fmt.Println("                           RESOURCE USAGE STATISTICS")
	fmt.Println("===================================================================================")
	fmt.Printf("Samples Collected: %d\n", rs.SampleCount)
	fmt.Println("-----------------------------------------------------------------------------------")
	fmt.Println("CPU Usage (cores):")
	fmt.Printf("  Min:     %.3f\n", rs.MinCPUCores)
	fmt.Printf("  Average: %.3f\n", rs.AvgCPUCores)
	fmt.Printf("  Max:     %.3f\n", rs.MaxCPUCores)
	fmt.Println("-----------------------------------------------------------------------------------")
	fmt.Println("Memory Usage (MB):")
	fmt.Printf("  Min:     %.2f\n", rs.MinMemoryMB)
	fmt.Printf("  Average: %.2f\n", rs.AvgMemoryMB)
	fmt.Printf("  Max:     %.2f\n", rs.MaxMemoryMB)
	fmt.Println("===================================================================================")
}
