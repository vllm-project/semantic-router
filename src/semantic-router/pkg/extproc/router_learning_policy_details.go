package extproc

import "github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"

type routerLearningPolicyDetails struct {
	SessionAware    *sessionAwareLearningDiagnostics
	Bandit          *routerLearningBanditDiagnostics
	Elo             *routerLearningEloDiagnostics
	Personalization *routerLearningPersonalizationDiagnostics
}

func (d routerLearningPolicyDetails) Empty() bool {
	return d.SessionAware == nil &&
		d.Bandit == nil &&
		d.Elo == nil &&
		d.Personalization == nil
}

func (d routerLearningPolicyDetails) ActiveMethods() []routerLearningMethod {
	methods := []routerLearningMethod{}
	if d.SessionAware != nil {
		methods = append(methods, routerLearningMethodSessionAware)
	}
	if d.Bandit != nil {
		methods = append(methods, routerLearningMethodBandit)
	}
	if d.Elo != nil {
		methods = append(methods, routerLearningMethodElo)
	}
	if d.Personalization != nil {
		methods = append(methods, routerLearningMethodPersonalization)
	}
	return methods
}

func (d routerLearningPolicyDetails) SessionAwareTrace() *selection.SessionPolicyTrace {
	if d.SessionAware == nil {
		return nil
	}
	return d.SessionAware.trace
}

func (d routerLearningPolicyDetails) appendTo(fields *routerLearningPolicyFields) {
	if fields == nil {
		return
	}
	if d.SessionAware != nil {
		d.SessionAware.appendLearningPolicyFields(fields)
	}
	if d.Bandit != nil {
		d.Bandit.appendLearningPolicyFields(fields)
	}
	if d.Elo != nil {
		d.Elo.appendLearningPolicyFields(fields)
	}
	if d.Personalization != nil {
		d.Personalization.appendLearningPolicyFields(fields)
	}
}
