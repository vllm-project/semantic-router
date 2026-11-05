//go:build !windows

package main

import (
	"encoding/json"
	"fmt"
	"io"
	"os"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// configValidationReport is what -validate-config prints: the verdict of the
// Router's own load-time validation of one file, and the warnings it would
// log, without starting anything. The CLI's `vllm-sr config validate` reads it.
type configValidationReport struct {
	Valid    bool                   `json:"valid"`
	Error    string                 `json:"error,omitempty"`
	Warnings []config.ConfigWarning `json:"warnings"`
	// Migrated is the file with its removed and renamed forms and its
	// deprecated model fields in the forms that replace them
	// (-migrate-config), whether or not the file itself loads; Kept are its
	// notes: what migrate rewrote (migration_change), the deprecated fields
	// that have no replacement and stay, and what an operator must do
	// (migration_action); MigrationError is why it could not be written.
	Migrated       string                 `json:"migrated,omitempty"`
	Kept           []config.ConfigWarning `json:"kept,omitempty"`
	MigrationError string                 `json:"migration_error,omitempty"`
}

// validateConfigFile validates the file as the Router loads it, with
// environment references left unresolved, and returns the exit status: 0 when
// it would load, 1 when it wouldn't.
func validateConfigFile(path string, gateway config.GatewayMode, migrate bool, out io.Writer) int {
	report := configValidationReport{Warnings: []config.ConfigWarning{}}
	// The file may be a pipe (-config /dev/stdin): it is read once.
	data, err := os.ReadFile(path)
	if err == nil {
		err = validateConfigInto(&report, data, gateway)
	}
	if err != nil {
		report.Error = err.Error()
	}
	report.Valid = err == nil
	if migrate && data != nil {
		migrateConfigInto(&report, data)
	}
	encoded, encodeErr := json.Marshal(report)
	if encodeErr != nil {
		fmt.Fprintf(os.Stderr, "encode the validation report: %v\n", encodeErr)
		return 1
	}
	fmt.Fprintln(out, string(encoded))
	if !report.Valid {
		return 1
	}
	return 0
}

// migrateConfigInto writes the file's migration into the report.
func migrateConfigInto(report *configValidationReport, data []byte) {
	migrated, kept, err := config.MigrateConfig(data)
	if err != nil {
		report.MigrationError = err.Error()
		return
	}
	report.Migrated, report.Kept = string(migrated), kept
}

func validateConfigInto(report *configValidationReport, data []byte, gateway config.GatewayMode) error {
	mode, err := config.ParseGatewayMode(string(gateway))
	if err != nil {
		return err
	}
	cfg, err := config.ParseYAMLBytesWithoutEnvExpansion(data)
	if err != nil {
		return err
	}
	report.Warnings = append(report.Warnings, config.Warnings(cfg)...)
	return config.ValidateGatewayCapabilities(cfg, mode)
}
