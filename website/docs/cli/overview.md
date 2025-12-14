# VSR CLI Overview

The `vsr` (vLLM Semantic Router) CLI is a unified command-line tool designed to simplify the installation, configuration, deployment, and management of the Semantic Router.

## Key Features

- **Easy Installation**: Guided installation and setup process.
- **Configuration Management**: View, edit, validate, and modify configuration files with ease.
- **Deployment**: Deploy the router locally, via Docker Compose, or to Kubernetes with a single command.
- **Status & Monitoring**: Check service health and view logs.
- **Testing**: interactive prompt testing to verify routing logic.

## Installation

### From Binary

Download the latest release for your platform and add it to your PATH.

### From Source

```bash

make install-cli

```

## Quick Start

1. **Initialize a new configuration:**
   
   ```bash
   vsr init
   ```

   This creates a `config/config.yaml` file with default settings.

2. **Edit the configuration:**
   
   ```bash
   vsr config edit
   ```

   Opens the configuration file in your default editor.

3. **Validate the configuration:**
   
   ```bash
   vsr config validate
   ```

   Ensures your configuration is syntactically and semantically correct.

4. **Deploy the router:**
   
   ```bash
   vsr deploy docker
   ```

   Starts the router using Docker Compose.

5. **Check status:**
   
   ```bash
   vsr status
   ```

6. **Test a prompt:**
   
   ```bash
   vsr test-prompt "What is the derivative of x^2?"
   ```
   
## Next Steps

- [Command Reference](commands-reference.md)
- [Troubleshooting](troubleshooting.md)
