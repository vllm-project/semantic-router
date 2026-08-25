package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"os"

	"github.com/redis/go-redis/v9"

	"github.com/vllm-project/semantic-router/src/semantic-router/internal/accesscapacity"
)

const redisURLEnvironment = "ACCESS_CAPACITY_REDIS_URL"

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func run() error {
	config := accesscapacity.DefaultConfig()
	flag.IntVar(&config.KeyCount, "keys", config.KeyCount, "number of independent API keys")
	flag.IntVar(&config.Replicas, "replicas", config.Replicas, "Router-equivalent runtime clients")
	flag.IntVar(&config.Concurrency, "concurrency", config.Concurrency, "concurrent workload workers")
	flag.IntVar(&config.RequestLimit, "request-limit", config.RequestLimit, "per-key global request limit")
	flag.DurationVar(&config.OperationTimeout, "timeout", config.OperationTimeout, "complete gate timeout")
	flag.DurationVar(&config.UsageDrainTimeout, "usage-drain-timeout", config.UsageDrainTimeout, "usage delivery drain timeout")
	flag.StringVar(&config.KeyPrefix, "key-prefix", config.KeyPrefix, "isolated Redis key prefix; generated when empty")
	flag.StringVar(&config.OutputRoot, "output-root", config.OutputRoot, "gitignored report root")
	flag.BoolVar(&config.KeepData, "keep-data", config.KeepData, "retain the isolated key prefix after the run")
	flag.DurationVar(&config.Thresholds.MaxAdmissionP99, "max-admission-p99", config.Thresholds.MaxAdmissionP99, "maximum admission p99")
	flag.DurationVar(&config.Thresholds.MaxUsageLagP99, "max-usage-lag-p99", config.Thresholds.MaxUsageLagP99, "maximum usage observation p99")
	flag.Float64Var(&config.Thresholds.MinProjectionKeysPerS, "min-projection-keys-per-second", config.Thresholds.MinProjectionKeysPerS, "minimum compile-and-publish throughput")
	flag.Int64Var(&config.Thresholds.MaxProjectionBytesKey, "max-projection-bytes-per-key", config.Thresholds.MaxProjectionBytesKey, "maximum Redis projection bytes per API key")
	flag.Int64Var(&config.Thresholds.MaxEventBytes, "max-event-bytes", config.Thresholds.MaxEventBytes, "maximum incremental Redis bytes per settled event")
	flag.Parse()

	redisURL := os.Getenv(redisURLEnvironment)
	if redisURL == "" {
		return fmt.Errorf("%s is required", redisURLEnvironment)
	}
	redisOptions, err := redis.ParseURL(redisURL)
	if err != nil {
		return fmt.Errorf("%s is invalid", redisURLEnvironment)
	}
	report, gateErr := accesscapacity.Run(context.Background(), redisOptions, config)
	directory, reportErr := accesscapacity.WriteReport(config.OutputRoot, report)
	if reportErr != nil {
		return errors.Join(gateErr, reportErr)
	}
	fmt.Printf("access capacity gate: %s\nreport: %s\n", report.Status, directory)
	if gateErr != nil {
		return fmt.Errorf("capacity workload failed; inspect the report")
	}
	if report.Status != "passed" {
		return fmt.Errorf("capacity thresholds failed; inspect the report")
	}
	return nil
}
