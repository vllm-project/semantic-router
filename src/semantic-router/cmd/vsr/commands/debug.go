package commands

import (
	"fmt"

	"github.com/spf13/cobra"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/cli"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/cli/debug"
)

// NewDebugCmd creates the debug command
func NewDebugCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "debug",
		Short: "Run interactive debugging session",
		Long: `Run a comprehensive debugging session to identify issues.

This command performs the following checks:
  - Prerequisites (kubectl, docker, helm, make, git)
  - Configuration file validation
  - Model availability and integrity
  - System resources (disk space, ports)
  - Network connectivity

Examples:
  # Run full diagnostics
  vsr debug

  # Run with custom config
  vsr debug --config /path/to/config.yaml

  # Run with custom models directory
  vsr debug --models-dir /path/to/models`,
		RunE: func(cmd *cobra.Command, args []string) error {
			configPath := cmd.Parent().Flag("config").Value.String()
			modelsDir, _ := cmd.Flags().GetString("models-dir")

			cli.Info("Starting interactive debug session...")
			cli.Info("This will check your environment, configuration, and resources.")
			cli.Info("")

			// Run full diagnostics
			report := debug.RunFullDiagnostics(configPath, modelsDir)

			// Display report
			debug.DisplayReport(report)

			// Check if there are critical failures
			hasCriticalFailures := false
			for _, results := range [][]debug.CheckResult{
				report.Prerequisites,
				report.Configuration,
				report.ModelStatus,
			} {
				for _, result := range results {
					if result.Status == "fail" && result.Severity == "critical" {
						hasCriticalFailures = true
						break
					}
				}
				if hasCriticalFailures {
					break
				}
			}

			if hasCriticalFailures {
				cli.Error("\n❌ Critical issues found. Please resolve them before proceeding.")
				return fmt.Errorf("critical diagnostic failures")
			}

			cli.Success("\n✅ Debug session complete!")
			return nil
		},
	}

	cmd.Flags().String("models-dir", "./models", "Models directory to check")

	return cmd
}

// NewHealthCmd creates the health command
func NewHealthCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "health",
		Short: "Check router health",
		Long: `Perform a quick health check of the router and its components.

This is a lightweight check that verifies:
  - Configuration validity
  - Model availability
  - System resources
  - Service connectivity

Examples:
  # Quick health check
  vsr health

  # Health check with custom config
  vsr health --config /path/to/config.yaml`,
		RunE: func(cmd *cobra.Command, args []string) error {
			configPath := cmd.Parent().Flag("config").Value.String()
			modelsDir := "./models"

			cli.Info("Running health check...")
			cli.Info("")

			// Run quick checks
			configResults := debug.CheckConfiguration(configPath)
			modelResults := debug.CheckModelStatus(modelsDir)
			resourceResults := debug.CheckResources()
			connectivityResults := debug.CheckConnectivity(nil)

			// Display results
			allPass := true

			// Config
			for _, result := range configResults {
				switch result.Status {
				case "fail":
					cli.Error(fmt.Sprintf("✗ %s: %s", result.Name, result.Message))
					allPass = false
				case "warn":
					cli.Warning(fmt.Sprintf("⚠ %s: %s", result.Name, result.Message))
				default:
					cli.Success(fmt.Sprintf("✓ %s", result.Name))
				}
			}

			// Models
			for _, result := range modelResults {
				switch result.Status {
				case "fail":
					cli.Error(fmt.Sprintf("✗ %s: %s", result.Name, result.Message))
					allPass = false
				case "warn":
					cli.Warning(fmt.Sprintf("⚠ %s: %s", result.Name, result.Message))
				default:
					cli.Success(fmt.Sprintf("✓ %s", result.Name))
				}
			}

			// Resources
			for _, result := range resourceResults {
				switch result.Status {
				case "fail":
					cli.Error(fmt.Sprintf("✗ %s: %s", result.Name, result.Message))
					allPass = false
				case "warn":
					cli.Warning(fmt.Sprintf("⚠ %s: %s", result.Name, result.Message))
				default:
					cli.Success(fmt.Sprintf("✓ %s", result.Name))
				}
			}

			// Connectivity
			hasConnectivity := false
			for _, result := range connectivityResults {
				switch result.Status {
				case "pass":
					cli.Success(fmt.Sprintf("✓ %s is reachable", result.Name))
					hasConnectivity = true
				case "warn":
					cli.Warning(fmt.Sprintf("⚠ %s: %s", result.Name, result.Message))
				default:
					// Don't fail on connectivity issues, just warn
					cli.Warning(fmt.Sprintf("⚠ %s is not reachable", result.Name))
				}
			}

			cli.Info("")

			// Overall status
			if allPass && hasConnectivity {
				cli.Success("🟢 Overall Health: GOOD")
				cli.Info("All systems operational")
			} else if allPass {
				cli.Warning("🟡 Overall Health: DEGRADED")
				cli.Info("Router is not running but environment is ready")
				cli.Info("Deploy with: vsr deploy [local|docker|kubernetes|helm]")
			} else {
				cli.Error("🔴 Overall Health: POOR")
				cli.Info("Critical issues detected")
				cli.Info("Run 'vsr debug' for detailed diagnostics")
				return fmt.Errorf("health check failed")
			}

			return nil
		},
	}

	return cmd
}

// NewDiagnoseCmd creates the diagnose command
func NewDiagnoseCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "diagnose",
		Short: "Generate diagnostic report",
		Long: `Generate a comprehensive diagnostic report for troubleshooting.

The report includes:
  - System information
  - Environment checks
  - Configuration status
  - Model status
  - Resource availability
  - Network connectivity
  - Recommendations

The report can be saved to a file for support tickets.

Examples:
  # Generate report to stdout
  vsr diagnose

  # Save report to file
  vsr diagnose --output report.txt

  # Generate with custom config
  vsr diagnose --config /path/to/config.yaml --output report.txt`,
		RunE: func(cmd *cobra.Command, args []string) error {
			configPath := cmd.Parent().Flag("config").Value.String()
			modelsDir, _ := cmd.Flags().GetString("models-dir")
			outputFile, _ := cmd.Flags().GetString("output")

			// Run diagnostics
			report := debug.RunFullDiagnostics(configPath, modelsDir)

			// Display to stdout
			if outputFile == "" {
				debug.DisplayReport(report)
			} else {
				// Save to file
				cli.Info(fmt.Sprintf("Generating diagnostic report to: %s", outputFile))

				// TODO: Implement file output
				// For now, display and inform user
				debug.DisplayReport(report)

				cli.Info(fmt.Sprintf("\n📄 Report would be saved to: %s", outputFile))
				cli.Info("Note: File output not yet implemented")
			}

			return nil
		},
	}

	cmd.Flags().String("models-dir", "./models", "Models directory to check")
	cmd.Flags().String("output", "", "Output file for the report")

	return cmd
}
