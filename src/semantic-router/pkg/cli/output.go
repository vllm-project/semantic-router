package cli

import (
	"encoding/json"
	"os"

	"github.com/fatih/color"
	"github.com/olekukonko/tablewriter"
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
