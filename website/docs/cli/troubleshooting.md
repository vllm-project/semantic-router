# VSR CLI Troubleshooting

## Common Issues

### "Config file not found"

**Error:** `failed to read config: open config/config.yaml: no such file or directory`

**Solution:**
Run `vsr init` to generate a configuration file, or specify the correct path using the `--config` flag.

### "Validation failed"

**Error:** `❌ Semantic validation failed: ...`

**Solution:**

- Check that all models referenced in `decisions` are defined in `model_config`.
- Ensure at least one category is defined.
- Verify YAML syntax indentation.

### "Docker command not found"

**Error:** `docker-compose not found`

**Solution:**
Ensure Docker and Docker Compose are installed and available in your system PATH.

### "Endpoint not reachable" during deployment

**Solution:**

- Check if the router process is running (`vsr status`).
- Verify that the port (default 8080) is not blocked by a firewall.
- If running in Docker, ensure ports are correctly mapped.

### "Unknown resource" in `vsr get`

**Solution:**
Valid resources are `models`, `categories`, `decisions`, and `endpoints`. Check your spelling.

## Debugging

Use the `--verbose` flag to see detailed logs and error traces:

```bash
vsr deploy docker --verbose
```
