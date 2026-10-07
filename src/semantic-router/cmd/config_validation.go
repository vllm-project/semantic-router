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
}

// validateConfigFile validates the file as the Router loads it, with
// environment references left unresolved, and returns the exit status: 0 when
// it would load, 1 when it wouldn't.
func validateConfigFile(path string, gateway config.GatewayMode, out io.Writer) int {
	report := configValidationReport{Warnings: []config.ConfigWarning{}}
	err := validateConfigInto(&report, path, gateway)
	if err != nil {
		report.Error = err.Error()
	}
	report.Valid = err == nil
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

func validateConfigInto(report *configValidationReport, path string, gateway config.GatewayMode) error {
	mode, err := config.ParseGatewayMode(string(gateway))
	if err != nil {
		return err
	}
	data, err := os.ReadFile(path)
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
