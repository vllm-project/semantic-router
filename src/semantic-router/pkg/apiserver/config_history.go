//go:build !windows

package apiserver

import (
	"fmt"
	"net/http"
	"regexp"
	"strconv"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/configsnapshot"
)

// snapshotVersionPattern matches a configuration snapshot version given where
// the API also accepts a backup timestamp.
var snapshotVersionPattern = regexp.MustCompile(`^[1-9][0-9]{0,18}$`)

// configLifecycle is the Router's configuration lifecycle, or nil when this
// API serves without one.
func (s *ClassificationAPIServer) configLifecycle() *configsnapshot.Manager {
	if s == nil || s.runtimeRegistry == nil {
		return nil
	}
	return s.runtimeRegistry.ConfigLifecycle()
}

// preserveReplacedDocument keeps the document a mutation replaces
// recoverable, and returns the version that names it and the directory whose
// backups this API trims. With the Router's lifecycle the history keeps the
// document, once: a document it recorded already is not written again, and
// the history trims itself.
func (s *ClassificationAPIServer) preserveReplacedDocument(
	sourcePath string,
	data []byte,
	source configVersionSource,
) (string, string, error) {
	lifecycle := s.configLifecycle()
	if lifecycle == nil {
		backupDir := configBackupDir(sourcePath)
		version := nextConfigVersion(backupDir, time.Now())
		return version, backupDir, recordConfigBackup(backupDir, version, data, source)
	}
	if len(data) == 0 {
		return time.Now().Format(configVersionTimestampLayout), "", nil
	}
	history := lifecycle.History()
	hash := configDocumentETagHash(data)
	if record, ok := history.ByHash(hash); ok && record.Document != nil {
		return record.ID, "", nil
	}
	record, err := history.Add(configsnapshot.Record{
		Hash: hash, Origin: configsnapshot.Origin{Source: configsnapshot.Source(source)}, Document: data,
	})
	if err != nil {
		return "", "", err
	}
	return record.ID, "", nil
}

// managementOrigin attributes a configuration change to the management
// request that makes it.
func managementOrigin(r *http.Request, source configsnapshot.Source) configsnapshot.Origin {
	origin := configsnapshot.Origin{Source: source}
	if r == nil {
		return origin
	}
	if principal, ok := managementPrincipalFromContext(r.Context()); ok {
		origin.Principal = principal.Role
		if principal.Anonymous {
			origin.Principal = "anonymous"
		}
	}
	if requestID, ok := r.Context().Value(managementRequestIDContextKey).(string); ok {
		origin.RequestID = requestID
	}
	return origin
}

// attributeConfigWrite names the origin of the update the Router makes when it
// loads the document with hash. It returns the function that withdraws the
// attribution when the write fails.
func (s *ClassificationAPIServer) attributeConfigWrite(hash string, origin configsnapshot.Origin) func() {
	lifecycle := s.configLifecycle()
	if lifecycle == nil || origin.Source == "" {
		return func() {}
	}
	return lifecycle.Attribute(hash, origin)
}

// rollbackTarget is the recorded document a rollback restores.
type rollbackTarget struct {
	id       string
	version  uint64
	document []byte
	config   *config.RouterConfig
}

// resolveRollbackTarget finds the recorded document a rollback names, by
// configuration version or backup ID.
func (s *ClassificationAPIServer) resolveRollbackTarget(
	w http.ResponseWriter,
	req routerConfigRollbackRequest,
	backupDir string,
) (rollbackTarget, bool) {
	lifecycle := s.configLifecycle()
	if lifecycle == nil {
		if req.ConfigVersion != 0 {
			s.writeErrorResponse(w, http.StatusNotFound, "VERSION_NOT_FOUND",
				fmt.Sprintf("Configuration version %d is not recorded by this Router", req.ConfigVersion))
			return rollbackTarget{}, false
		}
		data, cfg, ok := s.loadRollbackBackup(w, backupFilePath(backupDir, req.Version), req.Version)
		return rollbackTarget{id: req.Version, document: data, config: cfg}, ok
	}
	history := lifecycle.History()
	record, found := history.ByVersion(req.ConfigVersion)
	label := strconv.FormatUint(req.ConfigVersion, 10)
	if req.ConfigVersion == 0 {
		record, found = history.ByID(req.Version)
		label = req.Version
	}
	if !found {
		s.writeErrorResponse(w, http.StatusNotFound, "VERSION_NOT_FOUND", fmt.Sprintf("Backup version %s not found", label))
		return rollbackTarget{}, false
	}
	if record.Document == nil {
		s.writeErrorResponse(w, http.StatusConflict, "VERSION_HAS_NO_DOCUMENT", fmt.Sprintf(
			"Configuration version %s came from a source that recorded no document, such as Kubernetes resources; restore it at that source",
			label))
		return rollbackTarget{}, false
	}
	cfg, err := config.ParseYAMLBytes(record.Document)
	if err != nil {
		s.writeErrorResponse(w, http.StatusBadRequest, "BACKUP_INVALID", fmt.Sprintf("Backup config is invalid: %v", err))
		return rollbackTarget{}, false
	}
	return rollbackTarget{id: record.ID, version: record.Version, document: record.Document, config: cfg}, true
}

// historyVersionEntries lists the configuration history, newest first.
func (s *ClassificationAPIServer) historyVersionEntries(history *configsnapshot.History) []RouterConfigVersionEntry {
	active := s.activeConfigSnapshot()
	entries := []RouterConfigVersionEntry{}
	for _, record := range history.List() {
		entry := RouterConfigVersionEntry{
			Version:       record.ID,
			Timestamp:     record.RecordedAt.Local().Format("2006-01-02 15:04:05"),
			Source:        historyVersionSource(record.Origin.Source),
			ConfigVersion: record.Version,
			Hash:          record.Hash,
			RollbackOf:    record.Origin.RollbackOf,
			Active:        active != nil && record.Version != 0 && record.Version == active.Version(),
		}
		if record.Document != nil {
			entry.Filename = "config." + record.ID + ".yaml"
		}
		entries = append(entries, entry)
	}
	return entries
}

func historyVersionSource(source configsnapshot.Source) configVersionSource {
	if known, ok := knownConfigVersionSources[configVersionSource(source)]; ok {
		return known
	}
	return configVersionSourceUnknown
}
