package cli

import (
	"encoding/json"
	"fmt"
	"os"

	"github.com/fatih/color"
	"github.com/olekukonko/tablewriter"
	"golang.org/x/term"
	"gopkg.in/yaml.v3"
)

// Color functions for terminal output
var (
	successColor = color.New(color.FgGreen, color.Bold)
	errorColor   = color.New(color.FgRed, color.Bold)
	warningColor = color.New(color.FgYellow, color.Bold)
	infoColor    = color.New(color.FgCyan)
)

// Success prints a success message in green
func Success(msg string) {
	successColor.Println(msg)
}

// Error prints an error message in red
func Error(msg string) {
	errorColor.Println(msg)
}

// Warning prints a warning message in yellow
func Warning(msg string) {
	warningColor.Println(msg)
}

// Info prints an info message in cyan
func Info(msg string) {
	infoColor.Println(msg)
}

// IsTerminal returns true if stdin is attached to a terminal.
// Use this to check if interactive prompts are safe to use.
func IsTerminal() bool {
	return term.IsTerminal(int(os.Stdin.Fd()))
}

// ConfirmPrompt prompts the user for confirmation and returns true if they agree.
// Returns an error if stdin is not a terminal (non-interactive mode).
// Callers should use the --force flag to skip prompts in CI/CD environments.
func ConfirmPrompt(message string) (bool, error) {
	if !IsTerminal() {
		return false, fmt.Errorf("cannot prompt for confirmation in non-interactive mode; use --force flag")
	}
	fmt.Print(message)
	var response string
	_, _ = fmt.Scanln(&response)
	return response == "y" || response == "Y", nil
}

// PrintTable prints data in table format
func PrintTable(headers []string, rows [][]string) {
	table := tablewriter.NewWriter(os.Stdout)
	table.SetHeader(headers)
	table.SetAutoWrapText(false)
	table.SetAutoFormatHeaders(true)
	table.SetHeaderAlignment(tablewriter.ALIGN_LEFT)
	table.SetAlignment(tablewriter.ALIGN_LEFT)
	table.SetCenterSeparator("")
	table.SetColumnSeparator("")
	table.SetRowSeparator("")
	table.SetHeaderLine(false)
	table.SetBorder(false)
	table.SetTablePadding("\t")
	table.SetNoWhiteSpace(true)

	for _, row := range rows {
		table.Append(row)
	}

	table.Render()
}

// PrintJSON prints data in JSON format
func PrintJSON(v interface{}) error {
	encoder := json.NewEncoder(os.Stdout)
	encoder.SetIndent("", "  ")
	return encoder.Encode(v)
}

// PrintYAML prints data in YAML format
func PrintYAML(v interface{}) error {
	encoder := yaml.NewEncoder(os.Stdout)
	encoder.SetIndent(2)
	defer encoder.Close()
	return encoder.Encode(v)
}
