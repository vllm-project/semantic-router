package handlers

import (
	"crypto/sha256"
	"errors"
	"fmt"
	"path/filepath"
	"regexp"
	"strings"
)

const (
	openClawManagedLabelKey = "ai.vllm.semantic-router.openclaw.managed"
	openClawOwnerLabelKey   = "ai.vllm.semantic-router.openclaw.owner"
	openClawManagedLabel    = openClawManagedLabelKey + "=true"
)

var (
	errOpenClawResourceNotFound = errors.New("OpenClaw managed resource not found")
	errOpenClawResourceConflict = errors.New("OpenClaw resource is not owned by this Dashboard")
	openClawContainerIDPattern  = regexp.MustCompile(`^[a-f0-9]{64}$`)
)

type openClawOwnedResourceKind uint8

const (
	openClawContainerResource openClawOwnedResourceKind = iota
	openClawVolumeResource
)

type openClawResourceInspection struct {
	exists    bool
	owned     bool
	reference string
	name      string
}

// openClawResolvedContainer binds one live registry record to the immutable
// container ID returned by an ownership-checked runtime inspection. Callers
// must use reference, never entry.Name, for runtime lifecycle/inspect/log
// operations because container names can be released and captured.
type openClawResolvedContainer struct {
	entry     ContainerEntry
	reference string
}

// openClawOwnerID is a stable, non-secret instance identity. The ownership
// check protects the host Docker namespace from accidental name capture by a
// Dashboard request; it is not intended as authentication against Docker-root.
func (h *OpenClawHandler) openClawOwnerID() (string, error) {
	if h == nil || strings.TrimSpace(h.dataDir) == "" {
		return "", errors.New("OpenClaw data directory is unavailable")
	}
	absPath, err := filepath.Abs(h.dataDir)
	if err != nil {
		return "", fmt.Errorf("resolve OpenClaw data directory: %w", err)
	}
	canonicalPath, err := filepath.EvalSymlinks(absPath)
	if err != nil {
		return "", fmt.Errorf("resolve canonical OpenClaw data directory: %w", err)
	}
	digest := sha256.Sum256([]byte(filepath.Clean(canonicalPath)))
	return fmt.Sprintf("%x", digest[:]), nil
}

func openClawRuntimeResourceMissing(output string) bool {
	lower := strings.ToLower(strings.TrimSpace(output))
	return strings.Contains(lower, "no such container") ||
		strings.Contains(lower, "no such object") ||
		strings.Contains(lower, "no such volume")
}

func (h *OpenClawHandler) inspectOpenClawOwnedResource(
	kind openClawOwnedResourceKind,
	name string,
) (openClawResourceInspection, error) {
	ownerID, err := h.openClawOwnerID()
	if err != nil {
		return openClawResourceInspection{}, err
	}
	format := fmt.Sprintf(
		`{{.Id}}|{{.Name}}|{{with .Config.Labels}}{{index . %q}}{{end}}|{{with .Config.Labels}}{{index . %q}}{{end}}`,
		openClawManagedLabelKey,
		openClawOwnerLabelKey,
	)
	args := []string{"inspect", "-f", format, name}
	if kind == openClawVolumeResource {
		format = fmt.Sprintf(
			`{{.Name}}|{{with .Labels}}{{index . %q}}{{end}}|{{with .Labels}}{{index . %q}}{{end}}`,
			openClawManagedLabelKey,
			openClawOwnerLabelKey,
		)
		args = []string{"volume", "inspect", "-f", format, name}
	}
	output, inspectErr := h.containerCombinedOutput(args...)
	if inspectErr != nil {
		if openClawRuntimeResourceMissing(string(output)) {
			return openClawResourceInspection{}, nil
		}
		return openClawResourceInspection{}, fmt.Errorf("inspect OpenClaw managed resource: %w", inspectErr)
	}
	fields := strings.Split(strings.TrimSpace(string(output)), "|")
	if kind == openClawVolumeResource {
		if len(fields) != 3 {
			return openClawResourceInspection{exists: true}, nil
		}
		reference := strings.TrimSpace(fields[0])
		return openClawResourceInspection{
			exists:    true,
			owned:     reference == name && strings.EqualFold(strings.TrimSpace(fields[1]), "true") && strings.TrimSpace(fields[2]) == ownerID,
			reference: reference,
			name:      reference,
		}, nil
	}
	if len(fields) != 4 {
		return openClawResourceInspection{exists: true}, nil
	}
	reference := strings.TrimSpace(fields[0])
	resourceName := strings.TrimPrefix(strings.TrimSpace(fields[1]), "/")
	structurallyValid := openClawContainerIDPattern.MatchString(reference) && resourceName != ""
	if !openClawContainerIDPattern.MatchString(name) {
		structurallyValid = structurallyValid && resourceName == name
	}
	return openClawResourceInspection{
		exists:    true,
		owned:     structurallyValid && strings.EqualFold(strings.TrimSpace(fields[2]), "true") && strings.TrimSpace(fields[3]) == ownerID,
		reference: reference,
		name:      resourceName,
	}, nil
}

// resolveRegisteredOwnedContainerLocked must be called while h.mu is held.
// Keeping registry lookup and ownership inspection behind the same helper
// prevents individual call sites from accidentally trusting one without the
// other.
func (h *OpenClawHandler) resolveRegisteredOwnedContainerLocked(name string) (openClawResolvedContainer, error) {
	entries, err := h.loadRegistry()
	if err != nil {
		return openClawResolvedContainer{}, fmt.Errorf("load OpenClaw registry: %w", err)
	}
	entryIndex := findContainerIndex(entries, name)
	if entryIndex < 0 {
		return openClawResolvedContainer{}, errOpenClawResourceNotFound
	}

	resolved := openClawResolvedContainer{entry: entries[entryIndex]}
	inspection, err := h.inspectOpenClawOwnedResource(openClawContainerResource, name)
	if err != nil {
		return resolved, err
	}
	if !inspection.exists {
		return resolved, errOpenClawResourceNotFound
	}
	if !inspection.owned {
		return resolved, errOpenClawResourceConflict
	}
	resolved.reference = inspection.reference
	return resolved, nil
}

func (h *OpenClawHandler) resolveRegisteredOwnedContainer(name string) (openClawResolvedContainer, error) {
	h.mu.RLock()
	defer h.mu.RUnlock()
	return h.resolveRegisteredOwnedContainerLocked(name)
}

func (h *OpenClawHandler) runRegisteredOwnedContainerAction(
	name string,
	action string,
) (openClawResolvedContainer, error) {
	switch action {
	case "start", "stop", "restart":
	default:
		return openClawResolvedContainer{}, errors.New("unsupported OpenClaw container action")
	}

	h.mu.Lock()
	defer h.mu.Unlock()
	resolved, err := h.resolveRegisteredOwnedContainerLocked(name)
	if err != nil {
		return resolved, err
	}
	if err := h.containerRun(action, resolved.reference); err != nil {
		return resolved, fmt.Errorf("run OpenClaw container action: %w", err)
	}
	return resolved, nil
}

func (h *OpenClawHandler) removeOwnedContainerIfPresent(name string) error {
	inspection, err := h.inspectOpenClawOwnedResource(openClawContainerResource, name)
	if err != nil {
		return err
	}
	if !inspection.exists {
		return nil
	}
	if !inspection.owned {
		return errOpenClawResourceConflict
	}
	if err := h.containerRun("rm", "-f", inspection.reference); err != nil {
		return fmt.Errorf("remove OpenClaw managed container: %w", err)
	}
	return nil
}

// verifyProvisionedOwnedContainer binds the successful `docker run` result to
// one immutable container ID before registry state is committed. All cleanup
// after creation uses that ID rather than a mutable Docker name.
func (h *OpenClawHandler) verifyProvisionedOwnedContainer(name, reference string) error {
	if !openClawContainerIDPattern.MatchString(reference) {
		return errOpenClawResourceConflict
	}
	inspection, err := h.inspectOpenClawOwnedResource(openClawContainerResource, reference)
	if err != nil {
		return err
	}
	if !inspection.exists {
		return errOpenClawResourceNotFound
	}
	if !inspection.owned || inspection.reference != reference || inspection.name != name {
		return errOpenClawResourceConflict
	}
	return nil
}

func (h *OpenClawHandler) removeProvisionedOwnedContainer(name, reference string) error {
	if err := h.verifyProvisionedOwnedContainer(name, reference); err != nil {
		return err
	}
	if err := h.containerRun("rm", "-f", reference); err != nil {
		return fmt.Errorf("remove OpenClaw managed container: %w", err)
	}
	return nil
}

func (h *OpenClawHandler) ensureOwnedVolume(name string) error {
	inspection, err := h.inspectOpenClawOwnedResource(openClawVolumeResource, name)
	if err != nil {
		return err
	}
	if inspection.exists {
		if !inspection.owned {
			return errOpenClawResourceConflict
		}
		return nil
	}
	ownerID, err := h.openClawOwnerID()
	if err != nil {
		return err
	}
	if _, createErr := h.containerCombinedOutput(
		"volume", "create",
		"--label", openClawManagedLabel,
		"--label", openClawOwnerLabelKey+"="+ownerID,
		name,
	); createErr != nil {
		return errors.New("create OpenClaw managed volume")
	}
	inspection, err = h.inspectOpenClawOwnedResource(openClawVolumeResource, name)
	if err != nil {
		return err
	}
	if !inspection.exists || !inspection.owned {
		return errOpenClawResourceConflict
	}
	return nil
}

func (h *OpenClawHandler) removeOwnedVolumeIfPresent(name string) error {
	inspection, err := h.inspectOpenClawOwnedResource(openClawVolumeResource, name)
	if err != nil {
		return err
	}
	if !inspection.exists {
		return nil
	}
	if !inspection.owned {
		return errOpenClawResourceConflict
	}
	if err := h.containerRun("volume", "rm", inspection.reference); err != nil {
		return fmt.Errorf("remove OpenClaw managed volume: %w", err)
	}
	return nil
}

func openClawLifecycleHTTPStatus(err error) int {
	switch {
	case errors.Is(err, errOpenClawResourceNotFound):
		return 404
	case errors.Is(err, errOpenClawResourceConflict):
		return 409
	default:
		return 500
	}
}

func openClawLifecyclePublicError(err error) string {
	switch {
	case errors.Is(err, errOpenClawResourceNotFound):
		return "OpenClaw managed container not found"
	case errors.Is(err, errOpenClawResourceConflict):
		return "container or volume name conflicts with a resource not owned by this Dashboard"
	default:
		return "OpenClaw container lifecycle operation failed"
	}
}
