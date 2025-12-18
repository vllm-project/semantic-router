package commands

import (
	"os"

	"github.com/spf13/cobra"
)

// NewCompletionCmd creates the completion command
func NewCompletionCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "completion [bash|zsh|fish|powershell]",
		Short: "Generate shell completion script",
		Long: `Generate shell completion script for VSR.

To load completions:

Bash:
  # Linux:
  $ vsr completion bash > /etc/bash_completion.d/vsr

  # macOS:
  $ vsr completion bash > /usr/local/etc/bash_completion.d/vsr

  # Current session:
  $ source <(vsr completion bash)

Zsh:
  # If shell completion is not already enabled:
  $ echo "autoload -U compinit; compinit" >> ~/.zshrc

  # Generate completion:
  $ vsr completion zsh > "${fpath[1]}/_vsr"

  # Current session:
  $ source <(vsr completion zsh)

Fish:
  $ vsr completion fish > ~/.config/fish/completions/vsr.fish

  # Current session:
  $ vsr completion fish | source

PowerShell:
  PS> vsr completion powershell | Out-String | Invoke-Expression

  # To load completions for every session:
  PS> vsr completion powershell > vsr.ps1
  # And source this file from your PowerShell profile.

Examples:
  # Generate bash completion
  vsr completion bash

  # Generate zsh completion and save to file
  vsr completion zsh > /usr/local/share/zsh/site-functions/_vsr

  # Generate fish completion
  vsr completion fish > ~/.config/fish/completions/vsr.fish

  # Generate PowerShell completion
  vsr completion powershell > vsr.ps1`,
		DisableFlagsInUseLine: true,
		ValidArgs:             []string{"bash", "zsh", "fish", "powershell"},
		Args:                  cobra.MatchAll(cobra.ExactArgs(1), cobra.OnlyValidArgs),
		RunE: func(cmd *cobra.Command, args []string) error {
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
			return nil
		},
	}

	return cmd
}
