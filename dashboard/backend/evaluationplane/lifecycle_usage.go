package evaluationplane

import (
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
)

type OwnerLifecycleUsage struct {
	PrincipalDigest string `json:"principal_digest"`
	RunCount        int    `json:"run_count"`
	CampaignCount   int    `json:"campaign_count"`
	HeldRuns        int    `json:"held_runs"`
	ProtectedRuns   int    `json:"protected_runs"`
	ActualBytes     int64  `json:"actual_bytes"`
	ReservedBytes   int64  `json:"reserved_bytes"`
	ChargeableBytes int64  `json:"chargeable_bytes"`
	MaxBytes        int64  `json:"max_bytes"`
	MaxRuns         int    `json:"max_runs"`
	MaxCampaigns    int    `json:"max_campaigns"`
}

type LifecycleUsageReport struct {
	SchemaVersion        string                `json:"schema_version"`
	PolicyRevision       string                `json:"policy_revision"`
	ManagedPhysicalBytes int64                 `json:"managed_physical_bytes"`
	ReservedBytes        int64                 `json:"reserved_bytes"`
	ChargeableBytes      int64                 `json:"chargeable_bytes"`
	MaxStoreBytes        int64                 `json:"max_store_bytes"`
	AuditBytes           int64                 `json:"audit_bytes"`
	MaxAuditBytes        int64                 `json:"max_audit_bytes"`
	RunCount             int                   `json:"run_count"`
	CampaignCount        int                   `json:"campaign_count"`
	Owners               []OwnerLifecycleUsage `json:"owners"`
}

type lifecycleUsageSnapshot struct {
	report LifecycleUsageReport
	owners map[string]OwnerLifecycleUsage
}

type lifecycleUsageAccumulator struct {
	store         *Store
	owners        map[string]OwnerLifecycleUsage
	ownerCAS      map[string]map[string]bool
	ownerReserved map[string]int64
	totalReserved int64
}

func (s *Store) Usage(actor Actor) (LifecycleUsageReport, error) {
	if err := validateActor(actor); err != nil {
		return LifecycleUsageReport{}, err
	}
	s.lifecycle.mu.Lock()
	defer s.lifecycle.mu.Unlock()
	s.lifecycle.evidenceMu.Lock()
	defer s.lifecycle.evidenceMu.Unlock()
	s.runIndex.coordinator.Lock()
	defer s.runIndex.coordinator.Unlock()
	s.mu.Lock()
	defer s.mu.Unlock()
	snapshot, err := s.lifecycleUsageUnlocked()
	if err != nil {
		return LifecycleUsageReport{}, err
	}
	if actor.administrator {
		return snapshot.report, nil
	}
	owner := snapshot.owners[actor.principalDigest]
	if owner.PrincipalDigest == "" {
		owner = OwnerLifecycleUsage{
			PrincipalDigest: actor.principalDigest,
			MaxBytes:        s.lifecyclePolicy.Limits.MaxOwnerBytes, MaxRuns: s.lifecyclePolicy.Limits.MaxOwnerRuns,
			MaxCampaigns: s.lifecyclePolicy.Limits.MaxOwnerCampaigns,
		}
	}
	report := LifecycleUsageReport{
		SchemaVersion:  snapshot.report.SchemaVersion,
		PolicyRevision: snapshot.report.PolicyRevision,
		Owners:         []OwnerLifecycleUsage{owner},
	}
	return report, nil
}

func (s *Store) lifecycleUsageUnlocked() (lifecycleUsageSnapshot, error) {
	runs, ledgerErr := s.loadCompleteRunReferenceLedgerUnlocked()
	if ledgerErr != nil {
		return lifecycleUsageSnapshot{}, fmt.Errorf("%w: lifecycle usage requires a complete run ledger: %w", ErrConflict, ledgerErr)
	}
	usage := &lifecycleUsageAccumulator{
		store: s, owners: make(map[string]OwnerLifecycleUsage),
		ownerCAS: make(map[string]map[string]bool), ownerReserved: make(map[string]int64),
	}
	for _, run := range runs {
		if err := usage.addRun(run); err != nil {
			return lifecycleUsageSnapshot{}, err
		}
	}
	if err := usage.addControlledPairs(); err != nil {
		return lifecycleUsageSnapshot{}, err
	}
	campaignCount, err := usage.addCampaigns()
	if err != nil {
		return lifecycleUsageSnapshot{}, err
	}
	if err := usage.addCASEvidence(); err != nil {
		return lifecycleUsageSnapshot{}, err
	}
	return usage.snapshot(len(runs), campaignCount)
}

func (usage *lifecycleUsageAccumulator) addRun(run Run) error {
	s := usage.store
	lifecycle, err := s.readRunLifecycle(run)
	if err != nil {
		return fmt.Errorf("%w: lifecycle usage requires valid ownership metadata", ErrConflict)
	}
	bytes, err := privateDirectoryBytes(filepath.Join(s.runsRoot, run.ID), "")
	if err != nil {
		return err
	}
	attestationBytes, err := s.executionAttestationBytes(run.ID)
	if err != nil {
		return err
	}
	bytes, err = checkedLifecycleBytes(bytes, attestationBytes)
	if err != nil {
		return err
	}
	owner := usage.owners[lifecycle.OwnerPrincipalDigest]
	owner.PrincipalDigest = lifecycle.OwnerPrincipalDigest
	owner.RunCount++
	owner.ActualBytes, err = checkedLifecycleBytes(owner.ActualBytes, bytes)
	if err != nil {
		return err
	}
	remainingReservation := s.lifecyclePolicy.ReservedRunBytes - bytes
	if remainingReservation < 0 {
		remainingReservation = 0
	}
	usage.ownerReserved[lifecycle.OwnerPrincipalDigest], err = checkedLifecycleBytes(
		usage.ownerReserved[lifecycle.OwnerPrincipalDigest], remainingReservation,
	)
	if err != nil {
		return err
	}
	usage.totalReserved, err = checkedLifecycleBytes(usage.totalReserved, remainingReservation)
	if err != nil {
		return err
	}
	if lifecycle.EvidenceHold {
		owner.HeldRuns++
	}
	if lifecycle.RetentionClass == RetentionProtected {
		owner.ProtectedRuns++
	}
	usage.owners[lifecycle.OwnerPrincipalDigest] = owner
	references := make(map[string]bool)
	if err := s.markRunCASReferences(run.ID, references); err != nil {
		return fmt.Errorf("%w: lifecycle usage cannot verify run evidence: %w", ErrConflict, err)
	}
	if usage.ownerCAS[lifecycle.OwnerPrincipalDigest] == nil {
		usage.ownerCAS[lifecycle.OwnerPrincipalDigest] = make(map[string]bool)
	}
	for digest := range references {
		usage.ownerCAS[lifecycle.OwnerPrincipalDigest][digest] = true
	}
	return nil
}

func (usage *lifecycleUsageAccumulator) addControlledPairs() error {
	s := usage.store
	pairEntries, err := os.ReadDir(s.controlledPairRoot)
	if err != nil {
		return fmt.Errorf("list controlled pair usage: %w", err)
	}
	for _, entry := range pairEntries {
		if !entry.IsDir() || !validClientRequestID(entry.Name()) {
			return fmt.Errorf("%w: controlled pair usage ledger is invalid", ErrConflict)
		}
		pair, err := s.readControlledPair(entry.Name())
		if err != nil {
			return err
		}
		pairBytes, err := privateDirectoryBytes(filepath.Join(s.controlledPairRoot, entry.Name()), "")
		if err != nil {
			return err
		}
		owner := usage.owners[pair.OwnerPrincipalDigest]
		owner.PrincipalDigest = pair.OwnerPrincipalDigest
		owner.ActualBytes, err = checkedLifecycleBytes(owner.ActualBytes, pairBytes)
		if err != nil {
			return err
		}
		if pair.State != controlledPairStateDeleted {
			envelopeBytes, envelopeErr := controlledPairIntentReservationBytes(pair)
			if envelopeErr != nil {
				return envelopeErr
			}
			remainingReservation := envelopeBytes - pairBytes
			if remainingReservation < 0 {
				remainingReservation = 0
			}
			usage.ownerReserved[pair.OwnerPrincipalDigest], err = checkedLifecycleBytes(usage.ownerReserved[pair.OwnerPrincipalDigest], remainingReservation)
			if err != nil {
				return err
			}
			usage.totalReserved, err = checkedLifecycleBytes(usage.totalReserved, remainingReservation)
			if err != nil {
				return err
			}
		}
		usage.owners[pair.OwnerPrincipalDigest] = owner
	}
	return nil
}

func (usage *lifecycleUsageAccumulator) addCampaigns() (int, error) {
	campaigns, err := usage.store.loadStoredCampaignsUnlocked()
	if err != nil {
		return 0, fmt.Errorf("%w: lifecycle usage requires a valid campaign ledger", ErrConflict)
	}
	for _, campaign := range campaigns {
		lifecycle, err := usage.store.readCampaignLifecycle(campaign)
		if err != nil {
			return 0, fmt.Errorf("%w: lifecycle usage requires valid campaign ownership metadata", ErrConflict)
		}
		bytes, err := privateDirectoryBytes(filepath.Join(usage.store.campaignRoot, campaign.ID), "")
		if err != nil {
			return 0, err
		}
		owner := usage.owners[lifecycle.OwnerPrincipalDigest]
		owner.PrincipalDigest = lifecycle.OwnerPrincipalDigest
		owner.CampaignCount++
		owner.ActualBytes, err = checkedLifecycleBytes(owner.ActualBytes, bytes)
		if err != nil {
			return 0, err
		}
		usage.owners[lifecycle.OwnerPrincipalDigest] = owner
	}
	return len(campaigns), nil
}

func (usage *lifecycleUsageAccumulator) addCASEvidence() error {
	casRoot := filepath.Join(usage.store.root, "objects", "sha256")
	for ownerDigest, references := range usage.ownerCAS {
		owner := usage.owners[ownerDigest]
		for digest := range references {
			info, err := os.Lstat(filepath.Join(casRoot, digest))
			if os.IsNotExist(err) {
				continue
			}
			if err != nil || !info.Mode().IsRegular() || info.Mode()&os.ModeSymlink != 0 || info.Mode().Perm() != 0o600 {
				return fmt.Errorf("%w: lifecycle usage cannot verify CAS evidence", ErrConflict)
			}
			owner.ActualBytes, err = checkedLifecycleBytes(owner.ActualBytes, info.Size())
			if err != nil {
				return err
			}
		}
		usage.owners[ownerDigest] = owner
	}
	return nil
}

func (usage *lifecycleUsageAccumulator) snapshot(runCount, campaignCount int) (lifecycleUsageSnapshot, error) {
	s := usage.store
	ownerList := make([]OwnerLifecycleUsage, 0, len(usage.owners))
	for digest, owner := range usage.owners {
		owner.ReservedBytes = usage.ownerReserved[digest]
		var err error
		owner.ChargeableBytes, err = checkedLifecycleBytes(owner.ActualBytes, owner.ReservedBytes)
		if err != nil {
			return lifecycleUsageSnapshot{}, err
		}
		owner.MaxBytes, owner.MaxRuns = s.lifecyclePolicy.Limits.MaxOwnerBytes, s.lifecyclePolicy.Limits.MaxOwnerRuns
		owner.MaxCampaigns = s.lifecyclePolicy.Limits.MaxOwnerCampaigns
		usage.owners[digest] = owner
		ownerList = append(ownerList, owner)
	}
	sort.Slice(ownerList, func(i, j int) bool { return ownerList[i].PrincipalDigest < ownerList[j].PrincipalDigest })
	managed, err := privateDirectoryBytes(s.root, s.lifecycleAuditRoot)
	if err != nil {
		return lifecycleUsageSnapshot{}, err
	}
	collectionBytes, err := privateDirectoryBytes(s.collectionRoot, "")
	if err != nil {
		return lifecycleUsageSnapshot{}, err
	}
	if collectionBytes > lifecycleCollectionReservedBytes {
		return lifecycleUsageSnapshot{}, fmt.Errorf("%w: lifecycle collection exceeds its reserved capacity", ErrQuota)
	}
	storeReserved, err := checkedLifecycleBytes(
		usage.totalReserved,
		lifecycleCollectionReservedBytes-collectionBytes,
	)
	if err != nil {
		return lifecycleUsageSnapshot{}, err
	}
	chargeable, err := checkedLifecycleBytes(managed, storeReserved)
	if err != nil {
		return lifecycleUsageSnapshot{}, err
	}
	return lifecycleUsageSnapshot{
		report: LifecycleUsageReport{
			SchemaVersion: lifecyclePolicySchemaVersion, PolicyRevision: lifecyclePolicyRevision,
			ManagedPhysicalBytes: managed, ReservedBytes: storeReserved, ChargeableBytes: chargeable,
			MaxStoreBytes: s.lifecyclePolicy.Limits.MaxStoreBytes,
			AuditBytes:    s.lifecycle.bytes, MaxAuditBytes: s.lifecyclePolicy.Limits.MaxAuditBytes,
			RunCount: runCount, CampaignCount: campaignCount, Owners: ownerList,
		},
		owners: usage.owners,
	}, nil
}

func (s *Store) requireCampaignCreateQuotaUnlocked(actor Actor, bundleBytes int64) (string, error) {
	if bundleBytes < 1 {
		return "quota_owner_bytes", fmt.Errorf("%w: campaign quota requires a positive bundle size", ErrInvalid)
	}
	snapshot, err := s.lifecycleUsageUnlocked()
	if err != nil {
		return "quota_store_bytes", err
	}
	owner := snapshot.owners[actor.principalDigest]
	if owner.CampaignCount >= s.lifecyclePolicy.Limits.MaxOwnerCampaigns {
		return "quota_owner_campaigns", fmt.Errorf("%w: owner campaign count is at capacity", ErrQuota)
	}
	if bundleBytes > s.lifecyclePolicy.Limits.MaxOwnerBytes ||
		owner.ChargeableBytes > s.lifecyclePolicy.Limits.MaxOwnerBytes-bundleBytes {
		return "quota_owner_bytes", fmt.Errorf("%w: owner campaign byte capacity is full", ErrQuota)
	}
	if bundleBytes > s.lifecyclePolicy.Limits.MaxStoreBytes ||
		snapshot.report.ChargeableBytes > s.lifecyclePolicy.Limits.MaxStoreBytes-bundleBytes {
		return "quota_store_bytes", fmt.Errorf("%w: evaluation store byte capacity is full", ErrQuota)
	}
	return "", nil
}

func checkedLifecycleBytes(left, right int64) (int64, error) {
	const maxInt64 = int64(^uint64(0) >> 1)
	if left < 0 || right < 0 || left > maxInt64-right {
		return 0, fmt.Errorf("%w: evaluation lifecycle byte count overflows", ErrQuota)
	}
	return left + right, nil
}

func privateDirectoryBytes(root, excludedRoot string) (int64, error) {
	root = filepath.Clean(root)
	excludedRoot = filepath.Clean(excludedRoot)
	var total int64
	err := filepath.WalkDir(root, func(path string, entry fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		if excludedRoot != "." && path == excludedRoot {
			return filepath.SkipDir
		}
		info, err := entry.Info()
		if err != nil {
			return err
		}
		if info.Mode()&os.ModeSymlink != 0 {
			return fmt.Errorf("evaluation lifecycle usage refuses symbolic links")
		}
		if info.IsDir() {
			if info.Mode().Perm() != 0o700 {
				return fmt.Errorf("evaluation lifecycle usage requires private directories")
			}
			return nil
		}
		if !info.Mode().IsRegular() || info.Mode().Perm() != 0o600 {
			return fmt.Errorf("evaluation lifecycle usage requires private regular files")
		}
		if info.Size() < 0 || total > int64(^uint64(0)>>1)-info.Size() {
			return fmt.Errorf("evaluation lifecycle usage byte count overflow")
		}
		total += info.Size()
		return nil
	})
	if err != nil {
		return 0, fmt.Errorf("measure evaluation lifecycle usage: %w", err)
	}
	return total, nil
}

func (s *Store) requireCreateQuotaUnlocked(actor Actor, runCount int, aggregateBytes int64) (string, error) {
	if runCount < 1 || aggregateBytes < 0 {
		return "quota_owner_runs", fmt.Errorf("%w: create quota requires a positive run count", ErrInvalid)
	}
	snapshot, err := s.lifecycleUsageUnlocked()
	if err != nil {
		return "quota_store_bytes", err
	}
	owner := snapshot.owners[actor.principalDigest]
	if runCount > s.lifecyclePolicy.Limits.MaxOwnerRuns ||
		owner.RunCount > s.lifecyclePolicy.Limits.MaxOwnerRuns-runCount {
		return "quota_owner_runs", fmt.Errorf("%w: owner run count is at capacity", ErrQuota)
	}
	ownerGrowth, ok := checkedPositiveInt64Product(s.lifecyclePolicy.ReservedRunBytes, runCount)
	if !ok {
		return "quota_owner_bytes", fmt.Errorf("%w: owner byte reservation overflows", ErrQuota)
	}
	ownerGrowth, err = checkedLifecycleBytes(ownerGrowth, aggregateBytes)
	if err != nil {
		return "quota_owner_bytes", err
	}
	if ownerGrowth > s.lifecyclePolicy.Limits.MaxOwnerBytes ||
		owner.ChargeableBytes > s.lifecyclePolicy.Limits.MaxOwnerBytes-ownerGrowth {
		return "quota_owner_bytes", fmt.Errorf("%w: owner byte capacity is full", ErrQuota)
	}
	if ownerGrowth > s.lifecyclePolicy.Limits.MaxStoreBytes ||
		snapshot.report.ChargeableBytes > s.lifecyclePolicy.Limits.MaxStoreBytes-ownerGrowth {
		return "quota_store_bytes", fmt.Errorf("%w: evaluation store byte capacity is full", ErrQuota)
	}
	return "", nil
}

func checkedPositiveInt64Product(value int64, count int) (int64, bool) {
	if value < 0 || count < 1 {
		return 0, false
	}
	const maxInt64 = int64(^uint64(0) >> 1)
	if value != 0 && int64(count) > maxInt64/value {
		return 0, false
	}
	return value * int64(count), true
}

func (s *Store) requireEvidenceQuotaUnlocked(runID string, runBytes, logicalCASBytes, physicalCASBytes int64) error {
	run, err := s.getRunUnlocked(runID)
	if err != nil {
		return err
	}
	lifecycle, err := s.readRunLifecycle(run)
	if err != nil {
		return err
	}
	snapshot, err := s.lifecycleUsageUnlocked()
	if err != nil {
		return err
	}
	owner := snapshot.owners[lifecycle.OwnerPrincipalDigest]
	currentRunBytes, err := privateDirectoryBytes(filepath.Join(s.runsRoot, runID), "")
	if err != nil {
		return err
	}
	attestationBytes, err := s.executionAttestationBytes(runID)
	if err != nil {
		return err
	}
	currentRunBytes += attestationBytes
	remainingReservation := s.lifecyclePolicy.ReservedRunBytes - currentRunBytes
	if remainingReservation < 0 {
		remainingReservation = 0
	}
	runGrowth := runBytes - remainingReservation
	if runGrowth < 0 {
		runGrowth = 0
	}
	ownerGrowth := runGrowth + logicalCASBytes
	if owner.ChargeableBytes > s.lifecyclePolicy.Limits.MaxOwnerBytes-ownerGrowth {
		return fmt.Errorf("%w: owner evidence byte capacity is full", ErrQuota)
	}
	storeGrowth := runGrowth + physicalCASBytes
	if snapshot.report.ChargeableBytes > s.lifecyclePolicy.Limits.MaxStoreBytes-storeGrowth {
		return fmt.Errorf("%w: evaluation store evidence byte capacity is full", ErrQuota)
	}
	return nil
}

func (s *Store) executionAttestationBytes(runID string) (int64, error) {
	path := filepath.Join(s.attestationRoot, runID+".json")
	info, err := os.Lstat(path)
	if os.IsNotExist(err) {
		return 0, nil
	}
	if err != nil {
		return 0, err
	}
	if !info.Mode().IsRegular() || info.Mode()&os.ModeSymlink != 0 || info.Mode().Perm() != 0o600 ||
		info.Size() > maxExecutionAttestationBytes+1 {
		return 0, fmt.Errorf("%w: lifecycle usage cannot verify execution attestation", ErrConflict)
	}
	if _, err := s.readExecutionAttestationForDurableManifest(runID); err != nil {
		return 0, fmt.Errorf("%w: lifecycle usage cannot validate execution attestation", ErrConflict)
	}
	return info.Size(), nil
}
