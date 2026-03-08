# E2E Profile Selection Playbook

- Use [tools/agent/e2e-profile-map.yaml](../../../tools/agent/e2e-profile-map.yaml) as the source of truth
- Run `make agent-e2e-affected CHANGED_FILES="..."` to execute only affected local profiles
- Broad changes trigger:
  - local default profile: `ai-gateway`
  - CI full standard profile matrix
- Profile-local changes trigger only the matching local and CI profiles
