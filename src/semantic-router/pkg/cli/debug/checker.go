package debug

import (
	"fmt"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"syscall"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/cli"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// CheckResult represents the result of a check
type CheckResult struct {
	Name     string
	Status   string // "pass", "warn", "fail"
	Message  string
	Details  []string
	Severity string // "critical", "warning", "info"
}

// DiagnosticReport contains all diagnostic information
type DiagnosticReport struct {
	Timestamp       time.Time
	SystemInfo      SystemInfo
	Prerequisites   []CheckResult
	Configuration   []CheckResult
	ModelStatus     []CheckResult
	Resources       []CheckResult
	Connectivity    []CheckResult
	Recommendations []string
}

// SystemInfo contains system information
type SystemInfo struct {
	OS           string
	Architecture string
	GoVersion    string
	Hostname     string
	WorkingDir   string
}

// CheckPrerequisites checks all required tools
func CheckPrerequisites() []CheckResult {
	results := []CheckResult{}

	// Check Go
	if version := runtime.Version(); version != "" {
		results = append(results, CheckResult{
			Name:     "Go",
			Status:   "pass",
			Message:  fmt.Sprintf("Found: %s", version),
			Severity: "info",
		})
	} else {
		results = append(results, CheckResult{
			Name:     "Go",
			Status:   "fail",
			Message:  "Go not found",
			Severity: "critical",
		})
	}

	// Check kubectl
	results = append(results, checkCommand("kubectl", "kubectl version --client --short", false))

	// Check docker
	results = append(results, checkCommand("docker", "docker --version", false))

	// Check docker-compose
	dockerComposeResult := checkCommand("docker-compose", "docker-compose --version", false)
	if dockerComposeResult.Status != "pass" {
		// Try docker compose (v2)
		dockerComposeResult = checkCommand("docker-compose", "docker compose version", false)
	}
	results = append(results, dockerComposeResult)

	// Check helm
	results = append(results, checkCommand("helm", "helm version --short", false))

	// Check make
	results = append(results, checkCommand("make", "make --version", false))

	// Check git
	results = append(results, checkCommand("git", "git --version", false))

	return results
}

// checkCommand checks if a command exists and runs successfully
func checkCommand(name, command string, critical bool) CheckResult {
	parts := strings.Fields(command)
	//nolint:gosec // G204: Command is from internal prerequisite checks, not user input
	cmd := exec.Command(parts[0], parts[1:]...)

	output, err := cmd.CombinedOutput()
	if err != nil {
		severity := "warning"
		if critical {
			severity = "critical"
		}
		return CheckResult{
			Name:     name,
			Status:   "fail",
			Message:  fmt.Sprintf("Not found or not working: %v", err),
			Severity: severity,
		}
	}

	// Extract version from output
	outputStr := strings.TrimSpace(string(output))
	lines := strings.Split(outputStr, "\n")
	version := lines[0]
	if len(version) > 100 {
		version = version[:100] + "..."
	}

	return CheckResult{
		Name:     name,
		Status:   "pass",
		Message:  version,
		Severity: "info",
	}
}

// CheckConfiguration validates the configuration file
func CheckConfiguration(configPath string) []CheckResult {
	results := []CheckResult{}

	// Check if config file exists
	if _, err := os.Stat(configPath); os.IsNotExist(err) {
		results = append(results, CheckResult{
			Name:     "Config File",
			Status:   "fail",
			Message:  fmt.Sprintf("Configuration file not found: %s", configPath),
			Severity: "critical",
		})
		return results
	}

	results = append(results, CheckResult{
		Name:     "Config File",
		Status:   "pass",
		Message:  fmt.Sprintf("Found: %s", configPath),
		Severity: "info",
	})

	// Try to parse config
	cfg, err := config.Parse(configPath)
	if err != nil {
		results = append(results, CheckResult{
			Name:     "Config Parse",
			Status:   "fail",
			Message:  fmt.Sprintf("Failed to parse: %v", err),
			Severity: "critical",
			Details:  []string{"Check YAML syntax", "Verify all required fields are present"},
		})
		return results
	}

	results = append(results, CheckResult{
		Name:     "Config Parse",
		Status:   "pass",
		Message:  "Configuration parsed successfully",
		Severity: "info",
	})

	// Validate config
	if err := cli.ValidateConfig(cfg); err != nil {
		results = append(results, CheckResult{
			Name:     "Config Validation",
			Status:   "fail",
			Message:  fmt.Sprintf("Validation failed: %v", err),
			Severity: "critical",
			Details:  strings.Split(err.Error(), "\n"),
		})
	} else {
		results = append(results, CheckResult{
			Name:     "Config Validation",
			Status:   "pass",
			Message:  "Configuration is valid",
			Severity: "info",
		})
	}

	return results
}

// CheckModelStatus checks model availability
func CheckModelStatus(modelsDir string) []CheckResult {
	results := []CheckResult{}

	// Check if models directory exists
	if _, err := os.Stat(modelsDir); os.IsNotExist(err) {
		results = append(results, CheckResult{
			Name:     "Models Directory",
			Status:   "fail",
			Message:  fmt.Sprintf("Models directory not found: %s", modelsDir),
			Severity: "critical",
			Details:  []string{"Run: make download-models", "Or create the directory manually"},
		})
		return results
	}

	results = append(results, CheckResult{
		Name:     "Models Directory",
		Status:   "pass",
		Message:  fmt.Sprintf("Found: %s", modelsDir),
		Severity: "info",
	})

	// Count model files
	modelCount := 0
	_ = filepath.Walk(modelsDir, func(path string, info os.FileInfo, err error) error {
		if err == nil && !info.IsDir() {
			if strings.HasSuffix(path, ".bin") || strings.HasSuffix(path, ".safetensors") {
				modelCount++
			}
		}
		return nil
	})

	if modelCount == 0 {
		results = append(results, CheckResult{
			Name:     "Model Files",
			Status:   "warn",
			Message:  "No model files found",
			Severity: "warning",
			Details:  []string{"Models may not be downloaded", "Run: make download-models"},
		})
	} else {
		results = append(results, CheckResult{
			Name:     "Model Files",
			Status:   "pass",
			Message:  fmt.Sprintf("Found %d model file(s)", modelCount),
			Severity: "info",
		})
	}

	return results
}

// CheckResources checks system resources
func CheckResources() []CheckResult {
	results := []CheckResult{}

	// Check disk space
	cwd, _ := os.Getwd()
	var stat syscall.Statfs_t
	_ = syscall.Statfs(cwd, &stat)

	// Available space in bytes
	//nolint:gosec // G115: Block size is always positive, conversion is safe
	availableSpace := stat.Bavail * uint64(stat.Bsize)
	//nolint:gosec // G115: Block size is always positive, conversion is safe
	totalSpace := stat.Blocks * uint64(stat.Bsize)
	usedSpace := totalSpace - availableSpace
	usedPercent := float64(usedSpace) / float64(totalSpace) * 100

	diskStatus := "pass"
	diskSeverity := "info"
	if usedPercent > 90 {
		diskStatus = "warn"
		diskSeverity = "warning"
	} else if usedPercent > 95 {
		diskStatus = "fail"
		diskSeverity = "critical"
	}

	results = append(results, CheckResult{
		Name:     "Disk Space",
		Status:   diskStatus,
		Message:  fmt.Sprintf("%.1f%% used (%.2f GB available)", usedPercent, float64(availableSpace)/1024/1024/1024),
		Severity: diskSeverity,
	})

	// Check common ports
	commonPorts := []int{8080, 8801, 8700, 3000, 9090}
	usedPorts := []int{}

	for _, port := range commonPorts {
		if !isPortAvailable(port) {
			usedPorts = append(usedPorts, port)
		}
	}

	if len(usedPorts) > 0 {
		results = append(results, CheckResult{
			Name:     "Port Availability",
			Status:   "warn",
			Message:  fmt.Sprintf("%d port(s) in use: %v", len(usedPorts), usedPorts),
			Severity: "warning",
			Details:  []string{"These ports are commonly used by the router", "Check: netstat -tulpn | grep <port>"},
		})
	} else {
		results = append(results, CheckResult{
			Name:     "Port Availability",
			Status:   "pass",
			Message:  "All common ports available",
			Severity: "info",
		})
	}

	return results
}

// CheckConnectivity checks network connectivity
func CheckConnectivity(endpoints []string) []CheckResult {
	results := []CheckResult{}

	if len(endpoints) == 0 {
		endpoints = []string{
			"http://localhost:8080/health",
			"http://localhost:8080/metrics",
		}
	}

	for _, endpoint := range endpoints {
		result := checkEndpoint(endpoint)
		results = append(results, result)
	}

	return results
}

// checkEndpoint checks if an endpoint is reachable
func checkEndpoint(endpoint string) CheckResult {
	client := &http.Client{
		Timeout: 5 * time.Second,
	}

	resp, err := client.Get(endpoint)
	if err != nil {
		return CheckResult{
			Name:     endpoint,
			Status:   "fail",
			Message:  fmt.Sprintf("Not reachable: %v", err),
			Severity: "warning",
		}
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 200 && resp.StatusCode < 300 {
		return CheckResult{
			Name:     endpoint,
			Status:   "pass",
			Message:  fmt.Sprintf("Reachable (HTTP %d)", resp.StatusCode),
			Severity: "info",
		}
	}

	return CheckResult{
		Name:     endpoint,
		Status:   "warn",
		Message:  fmt.Sprintf("Reachable but returned HTTP %d", resp.StatusCode),
		Severity: "warning",
	}
}

// isPortAvailable checks if a port is available
func isPortAvailable(port int) bool {
	address := fmt.Sprintf("localhost:%d", port)
	conn, err := net.DialTimeout("tcp", address, 1*time.Second)
	if err != nil {
		return true // Port is available (connection failed)
	}
	conn.Close()
	return false // Port is in use
}

// GetSystemInfo returns system information
func GetSystemInfo() SystemInfo {
	hostname, _ := os.Hostname()
	cwd, _ := os.Getwd()

	return SystemInfo{
		OS:           runtime.GOOS,
		Architecture: runtime.GOARCH,
		GoVersion:    runtime.Version(),
		Hostname:     hostname,
		WorkingDir:   cwd,
	}
}

// GenerateRecommendations generates recommendations based on check results
func GenerateRecommendations(report *DiagnosticReport) []string {
	recommendations := []string{}

	// Check for failed prerequisites
	for _, result := range report.Prerequisites {
		if result.Status == "fail" {
			switch result.Name {
			case "kubectl":
				recommendations = append(recommendations, "Install kubectl: https://kubernetes.io/docs/tasks/tools/")
			case "docker":
				recommendations = append(recommendations, "Install Docker: https://docs.docker.com/get-docker/")
			case "docker-compose":
				recommendations = append(recommendations, "Install Docker Compose: https://docs.docker.com/compose/install/")
			case "helm":
				recommendations = append(recommendations, "Install Helm: https://helm.sh/docs/intro/install/")
			case "make":
				recommendations = append(recommendations, "Install make: apt-get install build-essential (Ubuntu) or brew install make (macOS)")
			}
		}
	}

	// Check for config issues
	for _, result := range report.Configuration {
		if result.Status == "fail" && result.Name == "Config File" {
			recommendations = append(recommendations, "Initialize configuration: vsr init")
		}
	}

	// Check for model issues
	for _, result := range report.ModelStatus {
		if result.Status == "fail" || result.Status == "warn" {
			recommendations = append(recommendations, "Download models: make download-models")
			break
		}
	}

	// Check for resource issues
	for _, result := range report.Resources {
		if result.Name == "Disk Space" && result.Status != "pass" {
			recommendations = append(recommendations, "Free up disk space or clean up unused models: vsr model remove <model-id>")
		}
		if result.Name == "Port Availability" && result.Status == "warn" {
			recommendations = append(recommendations, "Stop services using required ports or configure different ports in config.yaml")
		}
	}

	// If everything passes
	if len(recommendations) == 0 {
		recommendations = append(recommendations, "All checks passed! You're ready to deploy.")
		recommendations = append(recommendations, "Deploy with: vsr deploy [local|docker|kubernetes|helm]")
	}

	return recommendations
}

// RunFullDiagnostics runs all diagnostic checks
func RunFullDiagnostics(configPath, modelsDir string) *DiagnosticReport {
	report := &DiagnosticReport{
		Timestamp:  time.Now(),
		SystemInfo: GetSystemInfo(),
	}

	cli.Info("Running comprehensive diagnostics...")
	cli.Info("")

	// Prerequisites
	cli.Info("Checking prerequisites...")
	report.Prerequisites = CheckPrerequisites()

	// Configuration
	cli.Info("Checking configuration...")
	report.Configuration = CheckConfiguration(configPath)

	// Models
	cli.Info("Checking models...")
	report.ModelStatus = CheckModelStatus(modelsDir)

	// Resources
	cli.Info("Checking system resources...")
	report.Resources = CheckResources()

	// Connectivity
	cli.Info("Checking connectivity...")
	report.Connectivity = CheckConnectivity(nil)

	// Generate recommendations
	report.Recommendations = GenerateRecommendations(report)

	return report
}

// DisplayReport displays a diagnostic report
func DisplayReport(report *DiagnosticReport) {
	cli.Info("\n╔════════════════════════════════════════════════════════════════╗")
	cli.Info("║                    Diagnostic Report                          ║")
	cli.Info("╚════════════════════════════════════════════════════════════════╝")

	// System Info
	cli.Info("\n📋 System Information:")
	cli.Info(fmt.Sprintf("  OS: %s (%s)", report.SystemInfo.OS, report.SystemInfo.Architecture))
	cli.Info(fmt.Sprintf("  Go: %s", report.SystemInfo.GoVersion))
	cli.Info(fmt.Sprintf("  Hostname: %s", report.SystemInfo.Hostname))
	cli.Info(fmt.Sprintf("  Working Directory: %s", report.SystemInfo.WorkingDir))
	cli.Info(fmt.Sprintf("  Timestamp: %s", report.Timestamp.Format(time.RFC3339)))

	// Display each category
	displayCheckCategory("Prerequisites", report.Prerequisites)
	displayCheckCategory("Configuration", report.Configuration)
	displayCheckCategory("Models", report.ModelStatus)
	displayCheckCategory("Resources", report.Resources)
	displayCheckCategory("Connectivity", report.Connectivity)

	// Recommendations
	if len(report.Recommendations) > 0 {
		cli.Info("\n💡 Recommendations:")
		for i, rec := range report.Recommendations {
			cli.Info(fmt.Sprintf("  %d. %s", i+1, rec))
		}
	}

	// Summary
	totalChecks := len(report.Prerequisites) + len(report.Configuration) +
		len(report.ModelStatus) + len(report.Resources) + len(report.Connectivity)
	passedChecks := 0
	failedChecks := 0
	warningChecks := 0

	for _, results := range [][]CheckResult{
		report.Prerequisites,
		report.Configuration,
		report.ModelStatus,
		report.Resources,
		report.Connectivity,
	} {
		for _, result := range results {
			switch result.Status {
			case "pass":
				passedChecks++
			case "fail":
				failedChecks++
			case "warn":
				warningChecks++
			}
		}
	}

	cli.Info(fmt.Sprintf("\n📊 Summary: %d checks (%d passed, %d warnings, %d failed)",
		totalChecks, passedChecks, warningChecks, failedChecks))
}

// displayCheckCategory displays a category of checks
func displayCheckCategory(category string, results []CheckResult) {
	if len(results) == 0 {
		return
	}

	cli.Info(fmt.Sprintf("\n🔍 %s:", category))
	for _, result := range results {
		symbol := getStatusSymbol(result.Status)
		cli.Info(fmt.Sprintf("  %s %-25s %s", symbol, result.Name, result.Message))
		if len(result.Details) > 0 {
			for _, detail := range result.Details {
				cli.Info(fmt.Sprintf("      → %s", detail))
			}
		}
	}
}

// getStatusSymbol returns a symbol for the status
func getStatusSymbol(status string) string {
	switch status {
	case "pass":
		return "✓"
	case "fail":
		return "✗"
	case "warn":
		return "⚠"
	default:
		return "•"
	}
}
