package evaluationplane

func (s *Service) RunLifecycle(actor Actor, runID string) (RunLifecycleView, error) {
	release, err := s.beginOperation()
	if err != nil {
		return RunLifecycleView{}, err
	}
	defer release()
	return s.store.RunLifecycle(actor, runID)
}

func (s *Service) UpdateRunLifecycle(
	actor Actor,
	runID string,
	request UpdateLifecycleRequest,
) (RunLifecycleView, error) {
	release, err := s.beginOperation()
	if err != nil {
		return RunLifecycleView{}, err
	}
	defer release()
	return s.store.UpdateRunLifecycle(actor, runID, request)
}

func (s *Service) LifecycleUsage(actor Actor) (LifecycleUsageReport, error) {
	release, err := s.beginOperation()
	if err != nil {
		return LifecycleUsageReport{}, err
	}
	defer release()
	return s.store.Usage(actor)
}
