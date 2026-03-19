package modelresearch

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"
)

type ManagerConfig struct {
	BaseDir             string
	RepoRoot            string
	PythonPath          string
	DefaultAPIBase      string
	DefaultRequestModel string
	Platform            string
	HTTPClient          *http.Client
	CommandRunner       commandRunner
}

type StreamEvent struct {
	CampaignID string
	Event      CampaignEvent
	Terminal   bool
}

type Manager struct {
	mu                  sync.Mutex
	campaigns           map[string]*Campaign
	cancelFns           map[string]context.CancelFunc
	baseDir             string
	campaignsDir        string
	repoRoot            string
	pythonPath          string
	defaultAPIBase      string
	defaultRequestModel string
	platform            string
	httpClient          *http.Client
	runCommand          commandRunner
	events              chan StreamEvent
}

func NewManager(cfg ManagerConfig) (*Manager, error) {
	baseDir := cfg.BaseDir
	if baseDir == "" {
		baseDir = filepath.Join(".", ".vllm-sr", "model-research")
	}
	campaignsDir := filepath.Join(baseDir, "campaigns")
	if err := os.MkdirAll(campaignsDir, 0o755); err != nil {
		return nil, err
	}

	if cfg.DefaultRequestModel == "" {
		cfg.DefaultRequestModel = "MoM"
	}
	if cfg.PythonPath == "" {
		cfg.PythonPath = "python3"
	}

	manager := &Manager{
		campaigns:           make(map[string]*Campaign),
		cancelFns:           make(map[string]context.CancelFunc),
		baseDir:             baseDir,
		campaignsDir:        campaignsDir,
		repoRoot:            cfg.RepoRoot,
		pythonPath:          cfg.PythonPath,
		defaultAPIBase:      strings.TrimRight(cfg.DefaultAPIBase, "/"),
		defaultRequestModel: cfg.DefaultRequestModel,
		platform:            strings.ToLower(strings.TrimSpace(cfg.Platform)),
		httpClient:          cfg.HTTPClient,
		runCommand:          cfg.CommandRunner,
		events:              make(chan StreamEvent, 256),
	}
	if manager.httpClient == nil {
		manager.httpClient = newHTTPClient()
	}
	if manager.runCommand == nil {
		manager.runCommand = defaultCommandRunner
	}
	if err := manager.loadCampaigns(); err != nil {
		return nil, err
	}
	return manager, nil
}

func (m *Manager) Events() <-chan StreamEvent {
	return m.events
}

func (m *Manager) Recipes() RecipesResponse {
	runtimeModels, err := fetchRuntimeModelsInfo(m.httpClient, m.defaultAPIBase)
	if err != nil {
		runtimeModels = nil
	}
	return recipesResponse(m.defaultAPIBase, m.defaultRequestModel, m.effectivePlatform(), runtimeModels)
}

func (m *Manager) ListCampaigns() []*Campaign {
	m.mu.Lock()
	defer m.mu.Unlock()

	items := make([]*Campaign, 0, len(m.campaigns))
	for _, campaign := range m.campaigns {
		items = append(items, cloneCampaign(campaign))
	}
	sort.Slice(items, func(i, j int) bool {
		return items[i].CreatedAt.After(items[j].CreatedAt)
	})
	return items
}

func (m *Manager) GetCampaign(id string) *Campaign {
	m.mu.Lock()
	defer m.mu.Unlock()
	if campaign, ok := m.campaigns[id]; ok {
		return cloneCampaign(campaign)
	}
	return nil
}

func (m *Manager) StartCampaign(req CreateCampaignRequest) (*Campaign, error) {
	def, err := resolveRecipe(req.Target, req.GoalTemplate)
	if err != nil {
		return nil, err
	}

	if strings.TrimSpace(req.Name) == "" {
		req.Name = strings.ReplaceAll(def.Label, " ", "-")
	}
	if req.Budget.MaxTrials <= 0 {
		req.Budget.MaxTrials = 2
	}
	if req.SuccessThresholdPP <= 0 {
		req.SuccessThresholdPP = def.DefaultSuccessThresholdPP
	}

	apiBase := strings.TrimRight(firstNonEmpty(req.Overrides.APIBaseOverride, m.defaultAPIBase), "/")
	requestModel := firstNonEmpty(req.Overrides.RequestModelOverride, m.defaultRequestModel)

	runtimeModels, _ := fetchRuntimeModelsInfo(m.httpClient, apiBase)
	baseline := resolveBaseline(def, runtimeModels, requestModel)

	if err := m.validateRequestModel(apiBase, requestModel, strings.TrimSpace(req.Overrides.RequestModelOverride) != ""); err != nil {
		return nil, err
	}

	now := time.Now().UTC()
	campaignID := fmt.Sprintf("research-%d", now.UnixMilli())
	campaignDir := filepath.Join(m.campaignsDir, campaignID)
	if err := os.MkdirAll(campaignDir, 0o755); err != nil {
		return nil, err
	}

	campaign := &Campaign{
		ID:                  campaignID,
		Name:                req.Name,
		Status:              StatusPending,
		GoalTemplate:        req.GoalTemplate,
		Target:              req.Target,
		Platform:            m.resolveCampaignPlatform(req.Overrides.AllowCPUDryRun),
		PrimaryMetric:       def.PrimaryMetric,
		SuccessThresholdPP:  req.SuccessThresholdPP,
		Budget:              req.Budget,
		CreatedAt:           now,
		UpdatedAt:           now,
		DefaultAPIBase:      m.defaultAPIBase,
		APIBase:             apiBase,
		DefaultRequestModel: m.defaultRequestModel,
		RequestModel:        requestModel,
		Overrides:           req.Overrides,
		Recipe: RecipeSummary{
			Key:                         def.Key,
			Label:                       def.Label,
			GoalTemplates:               def.GoalTemplates,
			DefaultDataset:              def.DefaultDataset,
			DatasetHint:                 def.DatasetHint,
			DefaultSuccessThresholdPP:   def.DefaultSuccessThresholdPP,
			PrimaryMetric:               def.PrimaryMetric,
			SupportsDatasetOverride:     def.SupportsDatasetOverride,
			SupportsHyperparameterHints: true,
			Baseline:                    baseline,
		},
		Baseline:      baseline,
		ArtifactDir:   campaignDir,
		RuntimeModels: runtimeModels,
	}

	m.mu.Lock()
	m.campaigns[campaignID] = campaign
	m.mu.Unlock()

	m.recordEvent(campaignID, CampaignEvent{
		Timestamp: now,
		Kind:      EventStatus,
		Level:     "info",
		Message:   "Campaign created",
	})

	if !strings.EqualFold(m.platform, "amd") && !req.Overrides.AllowCPUDryRun {
		m.failCampaign(campaignID, StatusBlocked, "AMD platform is required unless CPU dry run is explicitly enabled")
		return m.GetCampaign(campaignID), nil
	}

	ctx, cancel := context.WithCancel(context.Background())
	m.mu.Lock()
	m.cancelFns[campaignID] = cancel
	m.mu.Unlock()

	go m.runCampaign(ctx, campaignID, def)
	return m.GetCampaign(campaignID), nil
}

func (m *Manager) StopCampaign(id string) error {
	m.mu.Lock()
	cancel, ok := m.cancelFns[id]
	m.mu.Unlock()
	if !ok {
		campaign := m.GetCampaign(id)
		if campaign == nil {
			return errors.New("campaign not found")
		}
		if campaign.Status == StatusCompleted || campaign.Status == StatusFailed || campaign.Status == StatusStopped || campaign.Status == StatusBlocked {
			return nil
		}
		return errors.New("campaign cannot be stopped")
	}

	cancel()
	m.recordEvent(id, CampaignEvent{
		Timestamp: time.Now().UTC(),
		Kind:      EventStatus,
		Level:     "warn",
		Message:   "Stop requested",
	})
	return nil
}

func (m *Manager) effectivePlatform() string {
	if strings.EqualFold(m.platform, "amd") {
		return "amd"
	}
	return "cpu"
}

func (m *Manager) resolveCampaignPlatform(allowCPUDryRun bool) string {
	if strings.EqualFold(m.platform, "amd") {
		return "amd"
	}
	if allowCPUDryRun {
		return "cpu-dry-run"
	}
	return "blocked"
}

func (m *Manager) validateRequestModel(apiBase, requestModel string, strict bool) error {
	models, err := fetchAvailableModels(m.httpClient, apiBase)
	if err != nil {
		if strict {
			return err
		}
		return nil
	}
	if len(models) == 0 {
		return nil
	}
	for _, model := range models {
		if strings.EqualFold(model, requestModel) {
			return nil
		}
	}
	return fmt.Errorf("request model %q is not exposed by %s", requestModel, apiBase)
}

func (m *Manager) campaignStatePath(id string) string {
	return filepath.Join(m.campaignsDir, id, "state.json")
}

func (m *Manager) saveCampaign(campaign *Campaign) {
	path := m.campaignStatePath(campaign.ID)
	_ = os.MkdirAll(filepath.Dir(path), 0o755)
	payload, err := json.MarshalIndent(campaign, "", "  ")
	if err != nil {
		return
	}
	tmpPath := path + ".tmp"
	if err := os.WriteFile(tmpPath, payload, 0o644); err != nil {
		return
	}
	_ = os.Rename(tmpPath, path)
}

func (m *Manager) loadCampaigns() error {
	matches, err := filepath.Glob(filepath.Join(m.campaignsDir, "*", "state.json"))
	if err != nil {
		return err
	}
	for _, match := range matches {
		payload, err := os.ReadFile(match)
		if err != nil {
			return err
		}
		var campaign Campaign
		if err := json.Unmarshal(payload, &campaign); err != nil {
			return err
		}
		if campaign.Status == StatusRunning || campaign.Status == StatusPending {
			campaign.Status = StatusFailed
			campaign.LastError = "dashboard restarted before campaign completion"
			now := time.Now().UTC()
			campaign.CompletedAt = &now
			campaign.UpdatedAt = now
			campaign.Events = append(campaign.Events, CampaignEvent{
				Timestamp: now,
				Kind:      EventStatus,
				Level:     "error",
				Message:   campaign.LastError,
			})
			m.saveCampaign(&campaign)
		}
		m.campaigns[campaign.ID] = &campaign
	}
	return nil
}

func cloneCampaign(campaign *Campaign) *Campaign {
	if campaign == nil {
		return nil
	}
	payload, err := json.Marshal(campaign)
	if err != nil {
		copied := *campaign
		return &copied
	}
	var copied Campaign
	if err := json.Unmarshal(payload, &copied); err != nil {
		plain := *campaign
		return &plain
	}
	return &copied
}

func (m *Manager) updateCampaign(id string, mutator func(*Campaign)) {
	m.mu.Lock()
	defer m.mu.Unlock()
	campaign, ok := m.campaigns[id]
	if !ok {
		return
	}
	mutator(campaign)
	campaign.UpdatedAt = time.Now().UTC()
	m.saveCampaign(campaign)
}

func (m *Manager) recordEvent(id string, event CampaignEvent) {
	m.updateCampaign(id, func(campaign *Campaign) {
		campaign.Events = append(campaign.Events, event)
		if len(campaign.Events) > 200 {
			campaign.Events = append([]CampaignEvent(nil), campaign.Events[len(campaign.Events)-200:]...)
		}
	})
	m.broadcast(StreamEvent{
		CampaignID: id,
		Event:      event,
		Terminal:   false,
	})
}

func (m *Manager) broadcast(update StreamEvent) {
	select {
	case m.events <- update:
	default:
	}
}

func (m *Manager) failCampaign(id string, status CampaignStatus, message string) {
	now := time.Now().UTC()
	m.updateCampaign(id, func(campaign *Campaign) {
		campaign.Status = status
		campaign.LastError = message
		campaign.CompletedAt = &now
	})
	event := CampaignEvent{
		Timestamp: now,
		Kind:      EventStatus,
		Level:     "error",
		Message:   message,
		Percent:   100,
	}
	m.recordEvent(id, event)
	m.markTerminal(id)
}

func (m *Manager) completeCampaign(id string, message string) {
	now := time.Now().UTC()
	m.updateCampaign(id, func(campaign *Campaign) {
		campaign.Status = StatusCompleted
		campaign.CompletedAt = &now
	})
	event := CampaignEvent{
		Timestamp: now,
		Kind:      EventStatus,
		Level:     "info",
		Message:   message,
		Percent:   100,
	}
	m.recordEvent(id, event)
	m.markTerminal(id)
}
