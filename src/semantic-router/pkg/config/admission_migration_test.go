package config

import "testing"

func TestApplyBatchConcurrencyMigration(t *testing.T) {
	cfg := &RouterConfig{}
	cfg.API.BatchClassification.MaxConcurrency = 8
	cfg.ModelAdmission = map[string]AdmissionConfig{
		"prompt_guard": {MaxConcurrency: 2, OnOverflow: "shed"},
	}

	applyBatchConcurrencyMigration(cfg)

	if got := cfg.ModelAdmission["prompt_guard"]; got.MaxConcurrency != 2 || got.OnOverflow != "shed" {
		t.Fatalf("explicit admission entry was overwritten: %+v", got)
	}
	if len(cfg.ModelAdmission) != len(admissionDeploymentKeys) {
		t.Fatalf("admission entries = %d, want %d", len(cfg.ModelAdmission), len(admissionDeploymentKeys))
	}
	migrated := cfg.ModelAdmission["domain_classifier"]
	if migrated.MaxConcurrency != 8 || migrated.MaxQueue != 8 || migrated.OnOverflow != "wait" {
		t.Fatalf("migrated entry = %+v, want wait-mode defaults", migrated)
	}
	if err := validateModelAdmissionContracts(cfg); err != nil {
		t.Fatalf("migrated entries must validate: %v", err)
	}
}

func TestApplyBatchConcurrencyMigrationNoop(t *testing.T) {
	cfg := &RouterConfig{}
	applyBatchConcurrencyMigration(cfg)
	if len(cfg.ModelAdmission) != 0 {
		t.Fatalf("admission entries = %+v, want none", cfg.ModelAdmission)
	}
}
