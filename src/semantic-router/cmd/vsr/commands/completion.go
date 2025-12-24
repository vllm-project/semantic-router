package commands

import (
	"os"

	"github.com/spf13/cobra"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/cli/completion"
)

// NewCompletionCmd creates the completion command
func NewCompletionCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "completion [bash|zsh|fish|powershell]",
		Short: "Generate or install shell completion scripts",
		Long: `Generate or install shell completion scripts for VSR.

Subcommands:
  install     Install completions automatically (recommended)
  uninstall   Remove installed completions
  status      Show installation status

Generate completion (manual installation):
  bash        Generate bash completion script
  zsh         Generate zsh completion script
  fish        Generate fish completion script
  powershell  Generate PowerShell completion script

Examples:
  # Install completions automatically (recommended)
  vsr completion install

  # Install for specific shell
  vsr completion install bash

  # Install system-wide (requires sudo)
  vsr completion install --system

  # Check installation status
  vsr completion status

  # Generate completion script manually
  vsr completion bash > /etc/bash_completion.d/vsr`,
		DisableFlagsInUseLine: true,
		ValidArgs:             []string{"bash", "zsh", "fish", "powershell"},
		Args:                  cobra.MatchAll(cobra.MaximumNArgs(1), cobra.OnlyValidArgs),
		RunE: func(cmd *cobra.Command, args []string) error {
			// Backward compatibility: if shell is provided, generate completion
			if len(args) == 1 {
				switch args[0] {
				case "bash":
					return cmd.Root().GenBashCompletion(os.Stdout)
				case "zsh":
					return cmd.Root().GenZshCompletion(os.Stdout)
				case "fish":
					return cmd.Root().GenFishCompletion(os.Stdout, true)
				case "powershell":
					return cmd.Root().GenPowerShellCompletion(os.Stdout)
				}
			}

			// Show help if no args
			return cmd.Help()
		},
	}

	// Add subcommands
	cmd.AddCommand(newCompletionInstallCmd())
	cmd.AddCommand(newCompletionUninstallCmd())
	cmd.AddCommand(newCompletionStatusCmd())

	return cmd
}

// newCompletionInstallCmd creates the install subcommand
func newCompletionInstallCmd() *cobra.Command {
	var systemWide bool
	var dryRun bool
	var force bool

	cmd := &cobra.Command{
		Use:   "install [bash|zsh|fish|powershell]",
		Short: "Install shell completions automatically",
		Long: `Install shell completions automatically.

If no shell is specified, the current shell will be auto-detected.

Examples:
  # Auto-detect shell and install
  vsr completion install

  # Install for bash
  vsr completion install bash

  # Install system-wide (requires sudo)
  vsr completion install --system

  # Preview installation without making changes
  vsr completion install --dry-run

  # Force overwrite existing completion
  vsr completion install --force`,
		ValidArgs: []string{"bash", "zsh", "fish", "powershell"},
		Args:      cobra.MaximumNArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			shell := ""
			if len(args) == 1 {
				shell = args[0]
			}

			installer := completion.NewInstaller(cmd.Root(), shell, systemWide, dryRun, force)
			return installer.Install()
		},
	}

	cmd.Flags().BoolVar(&systemWide, "system", false, "Install system-wide (requires root privileges)")
	cmd.Flags().BoolVar(&dryRun, "dry-run", false, "Show what would be done without making changes")
	cmd.Flags().BoolVar(&force, "force", false, "Overwrite existing completion file")

	return cmd
}

// newCompletionUninstallCmd creates the uninstall subcommand
func newCompletionUninstallCmd() *cobra.Command {
	var dryRun bool

	cmd := &cobra.Command{
		Use:   "uninstall [bash|zsh|fish|powershell]",
		Short: "Remove installed completions",
		Long: `Remove installed shell completions.

If no shell is specified, the current shell will be auto-detected.

Examples:
  # Auto-detect shell and uninstall
  vsr completion uninstall

  # Uninstall bash completions
  vsr completion uninstall bash

  # Preview uninstall without making changes
  vsr completion uninstall --dry-run`,
		ValidArgs: []string{"bash", "zsh", "fish", "powershell"},
		Args:      cobra.MaximumNArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			shell := ""
			if len(args) == 1 {
				shell = args[0]
			}

			installer := completion.NewInstaller(cmd.Root(), shell, false, dryRun, false)
			return installer.Uninstall()
		},
	}

	cmd.Flags().BoolVar(&dryRun, "dry-run", false, "Show what would be done without making changes")

	return cmd
}

// newCompletionStatusCmd creates the status subcommand
func newCompletionStatusCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "status [bash|zsh|fish|powershell]",
		Short: "Show completion installation status",
		Long: `Show the installation status of shell completions.

If no shell is specified, status for all shells will be shown.

Examples:
  # Show status for all shells
  vsr completion status

  # Show status for bash only
  vsr completion status bash`,
		ValidArgs: []string{"bash", "zsh", "fish", "powershell"},
		Args:      cobra.MaximumNArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			shell := ""
			if len(args) == 1 {
				shell = args[0]
			}

			installer := completion.NewInstaller(cmd.Root(), shell, false, false, false)
			return installer.ShowStatus()
		},
	}

	return cmd
}
