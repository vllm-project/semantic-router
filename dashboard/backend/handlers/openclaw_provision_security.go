package handlers

import (
	"errors"
	"fmt"
	"strings"
)

const (
	maximumOpenClawProvisionRequestBytes = 64 * 1024
	maximumOpenClawSkillContentBytes     = 1024 * 1024
)

var (
	errOpenClawProductionNetworkConfiguration = errors.New("production OpenClaw network configuration is unavailable")
	errOpenClawProductionNetworkSelection     = errors.New("production OpenClaw network selection is not allowed")
)

// resolveOpenClawProvisionToken deliberately rotates every production
// credential. Reusing an existing value would migrate a weak/caller-chosen
// development token into production. Development retains the compatibility
// behavior of reusing an existing token or accepting one for a new worker.
func (h *OpenClawHandler) resolveOpenClawProvisionToken(
	containerName string,
	requestedToken string,
	production bool,
) (string, error) {
	if production {
		return generateSecretToken(24)
	}
	if reused := h.gatewayTokenForContainer(containerName); reused != "" {
		return reused, nil
	}
	if requested := strings.TrimSpace(requestedToken); requested != "" {
		return requested, nil
	}
	return generateSecretToken(24)
}

// resolveOpenClawProvisionNetworkMode applies the deployment trust boundary.
// In production the orchestrator is the only authority that selects the
// network. An API caller may omit networkMode, send the legacy generic
// host/bridge UI value (which is normalized), or repeat the exact configured
// value.
func resolveOpenClawProvisionNetworkMode(
	requestedNetwork string,
	configuredNetwork string,
	production bool,
) (string, error) {
	configuredNetwork = strings.TrimSpace(configuredNetwork)
	if production {
		if configuredNetwork == "" {
			return "", errOpenClawProductionNetworkConfiguration
		}
		if err := validateOpenClawNetworkMode(configuredNetwork, true); err != nil {
			return "", fmt.Errorf("%w: configured network is not an isolated user-defined network", errOpenClawProductionNetworkConfiguration)
		}
		normalizedRequested := strings.ToLower(strings.TrimSpace(requestedNetwork))
		if normalizedRequested == "" || normalizedRequested == "host" || normalizedRequested == "bridge" {
			return configuredNetwork, nil
		}
		if requestedNetwork != configuredNetwork {
			return "", errOpenClawProductionNetworkSelection
		}
		return configuredNetwork, nil
	}

	if configuredNetwork != "" {
		normalizedRequested := strings.ToLower(strings.TrimSpace(requestedNetwork))
		if normalizedRequested == "" || normalizedRequested == "host" || normalizedRequested == "bridge" {
			return configuredNetwork, nil
		}
	}
	if strings.TrimSpace(requestedNetwork) == "" {
		return "host", nil
	}
	return requestedNetwork, nil
}

// ensureOpenClawProvisionNetwork keeps development's create-if-needed
// compatibility, but production may only consume a pre-provisioned network.
// The formatted inspect output is compared exactly so an error page, warning,
// or different returned object cannot be treated as success.
func (h *OpenClawHandler) ensureOpenClawProvisionNetwork(networkMode string, production bool) error {
	if !isBridgeNetwork(networkMode) {
		return nil
	}
	if !production {
		_, _ = h.containerCombinedOutput("network", "create", "--driver", "bridge", networkMode)
	}
	out, err := h.containerCombinedOutput("network", "inspect", "-f", "{{.Name}}", networkMode)
	if err != nil || strings.TrimSpace(string(out)) != networkMode {
		if production {
			return errOpenClawProductionNetworkConfiguration
		}
		return errors.New("OpenClaw network is unavailable")
	}
	return nil
}

func validateOpenClawProvisionRequest(req ProvisionRequest) error {
	if req.Container.GatewayPort < 0 || req.Container.GatewayPort > 65535 {
		return errors.New("gatewayPort must be zero or a valid TCP port")
	}
	if req.Container.ModelContextWindow < 0 || req.Container.ModelContextWindow > 16*1024*1024 {
		return errors.New("modelContextWindow is outside the supported range")
	}
	fields := []struct {
		value   string
		maximum int
		name    string
	}{
		{req.Identity.Name, 256, "identity.name"},
		{req.Identity.Emoji, 64, "identity.emoji"},
		{req.Identity.Role, 1024, "identity.role"},
		{req.Identity.Vibe, 2048, "identity.vibe"},
		{req.Identity.Principles, 16 * 1024, "identity.principles"},
		{req.Identity.Boundaries, 16 * 1024, "identity.boundaries"},
		{req.Identity.UserName, 256, "identity.userName"},
		{req.Identity.UserNotes, 16 * 1024, "identity.userNotes"},
		{req.Container.ContainerName, 256, "container.containerName"},
		{req.Container.AuthToken, 4096, "container.authToken"},
		{req.Container.ModelBaseURL, 4096, "container.modelBaseUrl"},
		{req.Container.ModelAPIKey, 8192, "container.modelApiKey"},
		{req.Container.ModelName, 512, "container.modelName"},
		{req.Container.MemoryBackend, 128, "container.memoryBackend"},
		{req.Container.MemoryBaseURL, 4096, "container.memoryBaseUrl"},
		{req.Container.VectorStore, 128, "container.vectorStore"},
		{req.Container.BaseImage, 512, "container.baseImage"},
		{req.Container.NetworkMode, 128, "container.networkMode"},
		{req.TeamID, 128, "teamId"},
		{req.RoleKind, 64, "roleKind"},
	}
	for _, field := range fields {
		if len([]byte(field.value)) > field.maximum {
			return fmt.Errorf("%s exceeds %d bytes", field.name, field.maximum)
		}
	}
	return nil
}

func parseOpenClawContainerID(stdout []byte) (string, error) {
	reference := strings.TrimSpace(string(stdout))
	if !openClawContainerIDPattern.MatchString(reference) {
		return "", errors.New("container runtime did not return one immutable container ID")
	}
	return reference, nil
}

func publicOpenClawProvisionResponse(production bool, response ProvisionResponse) ProvisionResponse {
	if !production {
		return response
	}
	return ProvisionResponse{Success: response.Success, Message: response.Message}
}

// fetchOpenClawSkillContentForProvision treats the selected image as untrusted
// even though a manager chose it. Skill extraction runs without network, host
// mounts, writeable rootfs, Linux capabilities, or an image-controlled
// entrypoint, and its persisted output has an independent hard ceiling.
func (h *OpenClawHandler) fetchOpenClawSkillContentForProvision(skillID, baseImage string) (string, error) {
	if !validOpenClawSkillID(skillID) {
		return "", errors.New("invalid OpenClaw skill ID")
	}
	for _, containerPath := range []string{
		"/app/skills/" + skillID + "/SKILL.md",
		"/app/extensions/" + skillID + "/SKILL.md",
	} {
		out, err := h.containerOutput(
			"run", "--rm",
			"--network", "none",
			"--read-only",
			"--user", "65534:65534",
			"--cap-drop", "ALL",
			"--security-opt", "no-new-privileges:true",
			"--pids-limit", "32",
			"--memory", "128m",
			"--cpus", "0.25",
			"--entrypoint", "cat",
			baseImage,
			containerPath,
		)
		if err != nil || len(out) == 0 {
			continue
		}
		if len(out) > maximumOpenClawSkillContentBytes {
			return "", errors.New("OpenClaw skill content exceeds the persistence limit")
		}
		return string(out), nil
	}

	skills, err := h.loadSkills()
	if err != nil {
		return "", errors.New("OpenClaw skill catalog is unavailable")
	}
	for _, skill := range skills {
		if skill.ID != skillID {
			continue
		}
		content := fmt.Sprintf(
			"---\nname: %s\ndescription: %q\nuser-invocable: true\n---\n\n# %s\n\n%s\n",
			skill.ID,
			skill.Description,
			skill.Name,
			skill.Description,
		)
		if len(content) > maximumOpenClawSkillContentBytes {
			return "", errors.New("OpenClaw skill content exceeds the persistence limit")
		}
		return content, nil
	}
	return "", nil
}
