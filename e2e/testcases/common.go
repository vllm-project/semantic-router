package testcases

import (
	"context"
	"fmt"
	"math/rand"
	"strings"
	"time"

	"k8s.io/client-go/kubernetes"

	"github.com/vllm-project/semantic-router/e2e/pkg/helpers"
	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

// setupServiceConnection sets up port forwarding to the configured service
// and returns the local port to use for HTTP requests and a cleanup function
func setupServiceConnection(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) (string, func(), error) {
	svcConfig := opts.ServiceConfig
	if svcConfig.LabelSelector == "" && svcConfig.Name == "" {
		return "", nil, fmt.Errorf("service configuration is required: either LabelSelector or Name must be provided")
	}

	// Get the service name
	var serviceName string
	if svcConfig.LabelSelector != "" {
		var err error
		serviceName, err = helpers.GetEnvoyServiceName(ctx, client, svcConfig.LabelSelector, opts.Verbose)
		if err != nil {
			return "", nil, fmt.Errorf("failed to get service by label selector: %w", err)
		}
	} else {
		serviceName = svcConfig.Name
	}

	if opts.Verbose {
		fmt.Printf("[Test] Using service: %s/%s\n", svcConfig.Namespace, serviceName)
	}

	// Start port forwarding
	stopFunc, err := helpers.StartPortForward(ctx, client, opts.RestConfig, svcConfig.Namespace, serviceName, svcConfig.PortMapping, opts.Verbose)
	if err != nil {
		return "", nil, fmt.Errorf("failed to start port forwarding: %w", err)
	}

	// Wait a bit for port forwarding to stabilize
	time.Sleep(2 * time.Second)

	// Extract local port from port mapping (e.g., "8080:80" -> "8080")
	portParts := strings.Split(svcConfig.PortMapping, ":")
	if len(portParts) != 2 {
		stopFunc() // Clean up before returning error
		return "", nil, fmt.Errorf("invalid port mapping format: %s (expected localPort:servicePort)", svcConfig.PortMapping)
	}

	return portParts[0], stopFunc, nil
}

// Random content generation for stress tests

var (
	// Question templates for variety
	questionTemplates = []string{
		"What is %d + %d?",
		"Calculate %d * %d",
		"What is the result of %d - %d?",
		"Solve: %d divided by %d",
		"What is %d%% of %d?",
		"If I have %d apples and buy %d more, how many do I have?",
		"What is the square root of %d?",
		"What is %d to the power of %d?",
		"How many days are in %d weeks?",
		"What is the average of %d and %d?",
	}

	topics = []string{
		"Explain the concept of machine learning",
		"What is the capital of France?",
		"How does photosynthesis work?",
		"What are the benefits of exercise?",
		"Describe the water cycle",
		"What is quantum computing?",
		"How do vaccines work?",
		"What causes climate change?",
		"Explain blockchain technology",
		"What is artificial intelligence?",
		"How does the internet work?",
		"What is the theory of relativity?",
		"Describe the solar system",
		"What is DNA?",
		"How do airplanes fly?",
	}

	tasks = []string{
		"Write a short poem about nature",
		"Summarize the main points of renewable energy",
		"List 5 programming languages",
		"Describe a typical day in your life",
		"Explain how to make a sandwich",
		"Give me 3 tips for better sleep",
		"What are the colors of the rainbow?",
		"Name 5 countries in Europe",
		"Describe your favorite hobby",
		"What are the main food groups?",
	}
)

// generateRandomContent generates random request content for stress testing
func generateRandomContent(requestID int) string {
	// Use requestID as seed for reproducibility within a test run
	r := rand.New(rand.NewSource(time.Now().UnixNano() + int64(requestID)))

	contentType := r.Intn(3)

	switch contentType {
	case 0:
		// Math question
		template := questionTemplates[r.Intn(len(questionTemplates))]
		num1 := r.Intn(100) + 1
		num2 := r.Intn(100) + 1
		return fmt.Sprintf("Request #%d: "+template, requestID, num1, num2)
	case 1:
		// General topic
		topic := topics[r.Intn(len(topics))]
		return fmt.Sprintf("Request #%d: %s", requestID, topic)
	default:
		// Task
		task := tasks[r.Intn(len(tasks))]
		return fmt.Sprintf("Request #%d: %s", requestID, task)
	}
}

// formatResponseHeaders formats HTTP response headers for logging
func formatResponseHeaders(headers map[string][]string) string {
	if len(headers) == 0 {
		return "  (no headers)"
	}

	var sb strings.Builder
	for key, values := range headers {
		for _, value := range values {
			sb.WriteString(fmt.Sprintf("  %s: %s\n", key, value))
		}
	}
	return sb.String()
}
