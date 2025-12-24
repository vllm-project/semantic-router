package completion

import (
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"time"

	"github.com/spf13/cobra"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/cli"
)

// Installer handles shell completion installation
type Installer struct {
	shell      string
	systemWide bool
	dryRun     bool
	force      bool
	rootCmd    *cobra.Command
}

// NewInstaller creates a new completion installer
func NewInstaller(rootCmd *cobra.Command, shell string, systemWide, dryRun, force bool) *Installer {
	return &Installer{
		shell:      shell,
		systemWide: systemWide,
		dryRun:     dryRun,
		force:      force,
		rootCmd:    rootCmd,
	}
}

// getPowerShellPaths returns platform-specific PowerShell completion paths
func getPowerShellPaths() []string {
	if runtime.GOOS == "windows" {
		userProfile := os.Getenv("USERPROFILE")
		if userProfile == "" {
			userProfile = os.Getenv("HOME") // Fallback
		}
		programFiles := os.Getenv("ProgramFiles")
		if programFiles == "" {
			programFiles = "C:\\Program Files" // Fallback
		}
		return []string{
			filepath.Join(userProfile, "Documents", "PowerShell", "vsr.ps1"),
			filepath.Join(programFiles, "PowerShell", "Modules", "vsr", "vsr.ps1"),
		}
	}
	// Linux/macOS paths
	return []string{
		filepath.Join(os.Getenv("HOME"), ".config", "powershell", "vsr.ps1"),
		"/usr/local/share/powershell/Modules/vsr/vsr.ps1",
	}
}

// completionPaths returns installation paths for each shell
// First path is user-local (no sudo), second is system-wide (requires sudo)
var completionPaths = map[string][]string{
	"bash": {
		filepath.Join(os.Getenv("HOME"), ".bash_completion.d", "vsr"),
		"/etc/bash_completion.d/vsr",
	},
	"zsh": {
		filepath.Join(os.Getenv("HOME"), ".zsh", "completion", "_vsr"),
		"/usr/local/share/zsh/site-functions/_vsr",
	},
	"fish": {
		filepath.Join(os.Getenv("HOME"), ".config", "fish", "completions", "vsr.fish"),
		"/usr/share/fish/vendor_completions.d/vsr.fish",
	},
}

// DetectShell detects the current shell from environment
func DetectShell() string {
	shell := os.Getenv("SHELL")
	if shell == "" {
		return ""
	}

	// Extract shell name from path
	shellName := filepath.Base(shell)

	// Normalize shell names
	switch shellName {
	case "bash", "sh":
		return "bash"
	case "zsh":
		return "zsh"
	case "fish":
		return "fish"
	case "pwsh", "powershell":
		return "powershell"
	default:
		return ""
	}
}

// Install installs shell completions
func (i *Installer) Install() error {
	// Detect shell if not specified
	if i.shell == "" {
		i.shell = DetectShell()
		if i.shell == "" {
			return fmt.Errorf("unable to detect shell. Please specify shell explicitly: vsr completion install [bash|zsh|fish|powershell]")
		}
		cli.Info(fmt.Sprintf("Detected shell: %s", i.shell))
	}

	// Get installation paths
	var paths []string
	var ok bool

	if i.shell == "powershell" {
		paths = getPowerShellPaths()
		ok = true
	} else {
		paths, ok = completionPaths[i.shell]
	}

	if !ok {
		return fmt.Errorf("unsupported shell: %s. Supported shells: bash, zsh, fish, powershell", i.shell)
	}

	// Determine which path to use
	var targetPath string
	if i.systemWide {
		targetPath = paths[1] // System-wide path
	} else {
		targetPath = paths[0] // User-local path
	}

	// Expand home directory
	if strings.HasPrefix(targetPath, "~") {
		targetPath = filepath.Join(os.Getenv("HOME"), targetPath[1:])
	}

	// Check if file already exists
	if _, err := os.Stat(targetPath); err == nil && !i.force {
		cli.Warning(fmt.Sprintf("Completion file already exists: %s", targetPath))
		cli.Info("Use --force to overwrite")
		return nil
	}

	if i.dryRun {
		cli.Info("[DRY RUN] Would install completion to: " + targetPath)
		i.showNextSteps(targetPath)
		return nil
	}

	// Create directory if needed
	targetDir := filepath.Dir(targetPath)
	if i.systemWide {
		// System-wide installation requires sudo
		cli.Warning("System-wide installation requires root privileges")
		if err := i.createDirWithSudo(targetDir); err != nil {
			return fmt.Errorf("failed to create directory: %w", err)
		}
	} else {
		// User-local installation
		err := os.MkdirAll(targetDir, 0o755)
		if err != nil {
			return fmt.Errorf("failed to create directory %s: %w", targetDir, err)
		}
	}

	// Generate completion script
	var completionScript strings.Builder
	var err error

	switch i.shell {
	case "bash":
		err = i.rootCmd.GenBashCompletion(&completionScript)
	case "zsh":
		err = i.rootCmd.GenZshCompletion(&completionScript)
	case "fish":
		err = i.rootCmd.GenFishCompletion(&completionScript, true)
	case "powershell":
		err = i.rootCmd.GenPowerShellCompletion(&completionScript)
	default:
		return fmt.Errorf("unsupported shell: %s", i.shell)
	}

	if err != nil {
		return fmt.Errorf("failed to generate completion script: %w", err)
	}

	// Write completion file
	if i.systemWide {
		if err := i.writeFileWithSudo(targetPath, completionScript.String()); err != nil {
			return fmt.Errorf("failed to write completion file: %w", err)
		}
	} else {
		if err := os.WriteFile(targetPath, []byte(completionScript.String()), 0o644); err != nil {
			return fmt.Errorf("failed to write completion file: %w", err)
		}
	}

	cli.Success(fmt.Sprintf("Installed completion to: %s", targetPath))
	i.showNextSteps(targetPath)

	return nil
}

// Uninstall removes installed completions
func (i *Installer) Uninstall() error {
	// Detect shell if not specified
	if i.shell == "" {
		i.shell = DetectShell()
		if i.shell == "" {
			return fmt.Errorf("unable to detect shell. Please specify shell explicitly")
		}
	}

	// Get installation paths
	var paths []string
	var ok bool

	if i.shell == "powershell" {
		paths = getPowerShellPaths()
		ok = true
	} else {
		paths, ok = completionPaths[i.shell]
	}

	if !ok {
		return fmt.Errorf("unsupported shell: %s", i.shell)
	}

	removed := false

	// Try to remove from both locations
	for _, path := range paths {
		// Expand home directory
		if strings.HasPrefix(path, "~") {
			path = filepath.Join(os.Getenv("HOME"), path[1:])
		}

		if _, err := os.Stat(path); err == nil {
			if i.dryRun {
				cli.Info(fmt.Sprintf("[DRY RUN] Would remove: %s", path))
				removed = true
			} else {
				if err := os.Remove(path); err != nil {
					cli.Warning(fmt.Sprintf("Failed to remove %s: %v", path, err))
				} else {
					cli.Success(fmt.Sprintf("Removed: %s", path))
					removed = true
				}
			}
		}
	}

	if !removed {
		cli.Info("No completion files found")
	}

	return nil
}

// ShowStatus shows the installation status
func (i *Installer) ShowStatus() error {
	// If shell specified, check only that shell
	shells := []string{"bash", "zsh", "fish", "powershell"}
	if i.shell != "" {
		shells = []string{i.shell}
	}

	cli.Info("Completion installation status:")
	fmt.Println()

	for _, shell := range shells {
		var paths []string
		var ok bool

		if shell == "powershell" {
			paths = getPowerShellPaths()
			ok = true
		} else {
			paths, ok = completionPaths[shell]
		}

		if !ok {
			continue
		}

		installed := false
		var installedPath string

		for _, path := range paths {
			// Expand home directory
			if strings.HasPrefix(path, "~") {
				path = filepath.Join(os.Getenv("HOME"), path[1:])
			}

			if _, err := os.Stat(path); err == nil {
				installed = true
				installedPath = path
				break
			}
		}

		if installed {
			fmt.Printf("  %s: ✓ installed (%s)\n", shell, installedPath)
		} else {
			fmt.Printf("  %s: ✗ not installed\n", shell)
		}
	}

	return nil
}

// createDirWithSudo creates a directory with sudo
func (i *Installer) createDirWithSudo(dir string) error {
	// 60 second timeout to allow for interactive password entry
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, "sudo", "mkdir", "-p", dir)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	cmd.Stdin = os.Stdin
	if err := cmd.Run(); err != nil {
		if errors.Is(err, context.DeadlineExceeded) {
			return fmt.Errorf("sudo command timed out after 60 seconds")
		}
		return err
	}
	return nil
}

// writeFileWithSudo writes a file with sudo using tee
func (i *Installer) writeFileWithSudo(path string, content string) error {
	// 60 second timeout to allow for interactive password entry
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, "sudo", "tee", path)
	cmd.Stdin = strings.NewReader(content)
	cmd.Stdout = nil // Suppress tee output
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		if errors.Is(err, context.DeadlineExceeded) {
			return fmt.Errorf("sudo command timed out after 60 seconds")
		}
		return err
	}
	return nil
}

// showNextSteps displays next steps after installation
func (i *Installer) showNextSteps(path string) {
	fmt.Println()
	cli.Info("Next steps:")

	switch i.shell {
	case "bash":
		fmt.Println("  1. Restart your shell, or run:")
		fmt.Println("     source " + path)
		fmt.Println("  2. Test completion:")
		fmt.Println("     vsr <TAB>")

	case "zsh":
		fmt.Println("  1. Ensure completion is enabled in ~/.zshrc:")
		fmt.Println("     autoload -U compinit; compinit")
		fmt.Println("  2. Restart your shell, or run:")
		fmt.Println("     exec zsh")
		fmt.Println("  3. Test completion:")
		fmt.Println("     vsr <TAB>")

	case "fish":
		fmt.Println("  1. Restart your shell, or run:")
		fmt.Println("     exec fish")
		fmt.Println("  2. Test completion:")
		fmt.Println("     vsr <TAB>")

	case "powershell":
		fmt.Println("  1. Add to your PowerShell profile:")
		fmt.Println("     . " + path)
		fmt.Println("  2. Restart PowerShell")
		fmt.Println("  3. Test completion:")
		fmt.Println("     vsr <TAB>")
	}
}
