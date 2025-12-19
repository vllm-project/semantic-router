# VSR CLI Command Reference

## Global Flags

- `--config, -c`: Path to the configuration file (default: `config/config.yaml`)
- `--verbose, -v`: Enable verbose output for debugging
- `--output, -o`: Output format (table, json, yaml) (default: `table`)

## Commands

### `vsr init`

Initialize a new configuration file.

**Usage:**

```bash
vsr init [flags]
```

**Flags:**

- `--output`: Output path for the configuration file (default: `config/config.yaml`)
- `--template`: Template to use: `default`, `minimal`, `full` (default: `default`)

### `vsr config`

Manage router configuration.

**Subcommands:**

- `view`: Display the current configuration.
- `edit`: Open configuration in your default editor (uses `$EDITOR`).
- `validate`: Validate configuration file syntax and semantics.
- `set <key> <value>`: Set a specific configuration value using dot notation.
- `get <key>`: Retrieve a specific configuration value.

**Examples:**

```bash
vsr config set bert_model.threshold 0.7
vsr config get default_model
```

### `vsr deploy`

Deploy the router to a target environment.

**Usage:**

```bash
vsr deploy [local|docker|kubernetes] [flags]
```

**Subcommands:**

- `local`: Run the router as a local process.
- `docker`: Deploy using Docker Compose.
- `kubernetes`: Deploy to a Kubernetes cluster.

**Flags:**

- `--observability`: Enable observability stack (Prometheus, Grafana, Jaeger).
- `--namespace` (Kubernetes only): Target namespace (default: `default`).

### `vsr undeploy`

Remove a deployment.

**Usage:**

```bash
vsr undeploy [local|docker|kubernetes]
```

### `vsr status`

Check the status of the router and its components.

**Usage:**

```bash
vsr status
```

### `vsr logs`

Fetch or stream logs from the router.

**Usage:**

```bash
vsr logs [flags]
```

**Flags:**

- `--follow, -f`: Follow log output.
- `--tail, -n`: Number of lines to show from the end (default: 100).

### `vsr get`

Retrieve information about configured resources.

**Usage:**

```bash
vsr get [models|categories|decisions|endpoints]
```

### `vsr test-prompt`

Send a test prompt to the router to verify classification.

**Usage:**

```bash
vsr test-prompt <text> [flags]
```

**Flags:**

- `--endpoint`: Router API endpoint (default: `http://localhost:8080/v1/classify`).
