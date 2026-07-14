package handlers

import "fmt"

func generateOpenClawBridgeComposeYAML(
	req ProvisionRequest,
	dataDir string,
	ownerID string,
	volumeName string,
	networkMode string,
	healthCommand string,
) string {
	return fmt.Sprintf(`services:
  openclaw:
    image: %q
    container_name: %q
    labels:
      %q: "true"
      %q: %q
    user: "0:0"
    cap_drop: ["ALL"]
    security_opt: ["no-new-privileges:true"]
    pids_limit: 512
    mem_limit: 4g
    cpus: 2.0
    networks:
      - %q
    volumes:
      - %q
      - %q
      - %q
    environment:
      OPENCLAW_CONFIG_PATH: /config/openclaw.json
      OPENCLAW_STATE_DIR: /state
    healthcheck:
      test: ["CMD-SHELL", %q]
      interval: 30s
      timeout: 5s
      start_period: 15s
      retries: 3
    command: ["node", "openclaw.mjs", "gateway", "--allow-unconfigured", "--bind", "lan"]
    restart: unless-stopped

networks:
  %q:
    external: true

volumes:
  %q:
    name: %q
    labels:
      %q: "true"
      %q: %q
`, req.Container.BaseImage, req.Container.ContainerName,
		openClawManagedLabelKey, openClawOwnerLabelKey, ownerID, networkMode,
		dataDir+"/workspace:/workspace", dataDir+"/openclaw.json:/config/openclaw.json:ro", volumeName+":/state",
		healthCommand,
		networkMode, volumeName, volumeName,
		openClawManagedLabelKey, openClawOwnerLabelKey, ownerID)
}

func generateOpenClawNetworkModeComposeYAML(
	req ProvisionRequest,
	dataDir string,
	ownerID string,
	volumeName string,
	networkMode string,
	healthCommand string,
) string {
	return fmt.Sprintf(`services:
  openclaw:
    image: %q
    container_name: %q
    labels:
      %q: "true"
      %q: %q
    user: "0:0"
    cap_drop: ["ALL"]
    security_opt: ["no-new-privileges:true"]
    pids_limit: 512
    mem_limit: 4g
    cpus: 2.0
    network_mode: %q
    volumes:
      - %q
      - %q
      - %q
    environment:
      OPENCLAW_CONFIG_PATH: /config/openclaw.json
      OPENCLAW_STATE_DIR: /state
    healthcheck:
      test: ["CMD-SHELL", %q]
      interval: 30s
      timeout: 5s
      start_period: 15s
      retries: 3
    command: ["node", "openclaw.mjs", "gateway", "--allow-unconfigured", "--bind", "lan"]
    restart: unless-stopped

volumes:
  %q:
    name: %q
    labels:
      %q: "true"
      %q: %q
`, req.Container.BaseImage, req.Container.ContainerName,
		openClawManagedLabelKey, openClawOwnerLabelKey, ownerID, networkMode,
		dataDir+"/workspace:/workspace", dataDir+"/openclaw.json:/config/openclaw.json:ro", volumeName+":/state",
		healthCommand,
		volumeName, volumeName,
		openClawManagedLabelKey, openClawOwnerLabelKey, ownerID)
}
