package extproc

import (
	"math"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/spendbudget"
)

type shadowSpend struct {
	pricing    config.ModelPricing
	currency   string
	usage      responseUsageMetrics
	estimate   float64
	accounted  float64
	costSource string
	dispatched bool
}

func (d *shadowDispatcher) admitShadowGroup(jobs []*shadowJob, budget config.ShadowBudgetConfig) bool {
	if jobs[0].context.Err() != nil {
		for _, job := range jobs {
			d.recordUnstarted(job)
		}
		return false
	}
	var remainingTokens int64
	if budget.MaxEstimatedTokens != nil {
		remainingTokens = int64(*budget.MaxEstimatedTokens)
	}
	totalCost := 0.0
	for _, job := range jobs {
		prepared, reason, err := d.prepareShadowCall(job)
		if err != nil {
			d.recordShadowBudgetRefusal(jobs, reason, err)
			return false
		}
		tokens, err := prepared.estimateTokens(job.engine)
		if err != nil {
			d.recordShadowBudgetRefusal(jobs, shadowReasonBudgetUnbounded, err)
			return false
		}
		if tokens.output == 0 {
			tokens.output = defaultSpendOutputTokens(job.routerConfig, job.model)
		}
		if budget.MaxEstimatedTokens != nil {
			input := int64(tokens.input)
			if tokens.output > remainingTokens-input {
				d.recordShadowBudgetRefusal(jobs, shadowReasonTokenExhausted, nil)
				return false
			}
			remainingTokens -= input + tokens.output
		}
		job.prepared = prepared
		if budget.MaxEstimatedCost != nil {
			spend, ok := estimateShadowSpend(job, tokens)
			if !ok {
				d.recordShadowBudgetRefusal(jobs, shadowReasonBudgetUnpriced, nil)
				return false
			}
			job.spend = spend
			totalCost += spend.estimate
		}
	}
	if limit := budget.MaxEstimatedCost; limit != nil && spendbudget.Round(*limit-totalCost) < 0 {
		d.recordShadowBudgetRefusal(jobs, shadowReasonBudgetExhausted, nil)
		return false
	}
	return true
}

func estimateShadowSpend(job *shadowJob, tokens shadowTokenEstimate) (*shadowSpend, bool) {
	currency := job.routerConfig.Budgets.EffectiveCurrency()
	pricing, ok := job.routerConfig.GetFullModelPricing(job.model)
	if !ok || normalizeReplayCurrency(pricing.Currency) != currency {
		return nil, false
	}
	estimate := estimateDispatchCost(tokens.input, tokens.output, pricing)
	if math.IsInf(estimate, 1) {
		return nil, false
	}
	return &shadowSpend{pricing: pricing, currency: currency, estimate: estimate, costSource: "not_dispatched"}, true
}

func (d *shadowDispatcher) recordShadowBudgetRefusal(jobs []*shadowJob, reason string, err error) {
	for _, job := range jobs {
		job.spend = nil
		d.record(job, d.failShadow(shadowResult{startedAt: d.now()}, reason, err))
	}
}

func (s *shadowSpend) account(result shadowResult) {
	if result.inputKnown && result.outputKnown {
		s.accounted = costForResponseUsage(s.usage, s.pricing)
		s.costSource = "usage"
	} else if s.dispatched {
		s.accounted = s.estimate
		s.costSource = "estimate"
	}
}
