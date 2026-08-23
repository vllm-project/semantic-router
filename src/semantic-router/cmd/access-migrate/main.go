// Command access-migrate applies the explicit forward-only managed control-
// plane schema. Router replicas never run migrations during startup.
package main

import (
	"context"
	"database/sql"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"
	"time"

	_ "github.com/lib/pq"

	controlpostgres "github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/postgres"
)

type options struct {
	dsnFile string
	dsnEnv  string
	timeout time.Duration
}

func main() {
	if err := run(os.Args[1:], os.Stdout); err != nil {
		fmt.Fprintln(os.Stderr, "access-migrate:", err)
		os.Exit(1)
	}
}

func run(args []string, output io.Writer) (returnErr error) {
	flags := flag.NewFlagSet("access-migrate", flag.ContinueOnError)
	flags.SetOutput(io.Discard)
	var opts options
	flags.StringVar(&opts.dsnFile, "dsn-file", "", "path to a file containing the PostgreSQL DSN")
	flags.StringVar(&opts.dsnEnv, "dsn-env", "", "environment variable containing the PostgreSQL DSN")
	flags.DurationVar(&opts.timeout, "timeout", 5*time.Minute, "whole migration deadline")
	if err := flags.Parse(args); err != nil {
		return err
	}
	if flags.NArg() != 0 {
		return errors.New("positional arguments are not supported")
	}
	if opts.timeout <= 0 || opts.timeout > time.Hour {
		return errors.New("timeout must be positive and no greater than 1h")
	}
	dsn, runErr := resolveDSN(opts, os.LookupEnv, os.ReadFile)
	if runErr != nil {
		return runErr
	}
	db, runErr := sql.Open("postgres", dsn)
	if runErr != nil {
		return errors.New("open PostgreSQL connection")
	}
	defer func() {
		if closeErr := db.Close(); closeErr != nil {
			returnErr = errors.Join(returnErr, fmt.Errorf("close PostgreSQL connection: %w", closeErr))
		}
	}()
	ctx, cancel := context.WithTimeout(context.Background(), opts.timeout)
	defer cancel()
	if err := db.PingContext(ctx); err != nil {
		return errors.New("connect to PostgreSQL")
	}
	if err := (controlpostgres.Migrator{DB: db}).Apply(ctx); err != nil {
		return err
	}
	migrations, runErr := controlpostgres.Migrations()
	if runErr != nil {
		return runErr
	}
	latest := int64(0)
	if len(migrations) > 0 {
		latest = migrations[len(migrations)-1].Version
	}
	_, runErr = fmt.Fprintf(output, "control-plane schema is current at version %d\n", latest)
	return runErr
}

func resolveDSN(
	opts options,
	lookupEnv func(string) (string, bool),
	readFile func(string) ([]byte, error),
) (string, error) {
	if (opts.dsnFile == "") == (opts.dsnEnv == "") {
		return "", errors.New("exactly one of --dsn-file or --dsn-env is required")
	}
	var value string
	if opts.dsnFile != "" {
		if !strings.HasPrefix(opts.dsnFile, "/") || strings.TrimSpace(opts.dsnFile) != opts.dsnFile {
			return "", errors.New("--dsn-file must be an absolute canonical path")
		}
		contents, err := readFile(opts.dsnFile)
		if err != nil {
			return "", errors.New("read PostgreSQL DSN file")
		}
		value = string(contents)
	} else {
		if strings.TrimSpace(opts.dsnEnv) != opts.dsnEnv || opts.dsnEnv == "" {
			return "", errors.New("--dsn-env must be a canonical environment variable name")
		}
		var ok bool
		value, ok = lookupEnv(opts.dsnEnv)
		if !ok {
			return "", errors.New("PostgreSQL DSN environment variable is not set")
		}
	}
	trimmed := strings.TrimSpace(value)
	if trimmed == "" {
		return "", errors.New("PostgreSQL DSN is empty")
	}
	return trimmed, nil
}
