package outputtokens

// Contributor is the typed seam for a future request-wide ledger bound (#2861).
type Contributor interface {
	UpperBound() (value int64, source string, ok bool)
}

// Sources are named optional upper bounds. Omitted sources do not participate.
type Sources struct {
	Client         *int64
	Plugin         *int64
	ModelRef       *int64
	AlgorithmStage *int64
	Ledger         Contributor
}

const (
	SourceClient         = "client"
	SourcePlugin         = "plugin"
	SourceModelRef       = "model_ref"
	SourceAlgorithmStage = "algorithm_stage"
	SourceLedger         = "ledger"
)

const (
	FallbackCodecUnsupported = "codec_unsupported"
	FallbackBlockedParam     = "blocked_param"
)

// Result is the strictest applicable bound and the source that produced it.
type Result struct {
	Effective *int64
	Source    string
	Fallback  string
}

// Compose returns the minimum of all present positive bounds. Request input
// may only narrow operator, model, plugin, algorithm, or ledger ceilings.
func Compose(sources Sources) Result {
	type bound struct {
		source string
		value  int64
	}
	bounds := make([]bound, 0, 5)
	add := func(source string, value *int64) {
		if value != nil && *value >= 1 {
			bounds = append(bounds, bound{source: source, value: *value})
		}
	}
	add(SourceClient, sources.Client)
	add(SourcePlugin, sources.Plugin)
	add(SourceModelRef, sources.ModelRef)
	add(SourceAlgorithmStage, sources.AlgorithmStage)
	if sources.Ledger != nil {
		if value, source, ok := sources.Ledger.UpperBound(); ok && value >= 1 {
			if source == "" {
				source = SourceLedger
			}
			bounds = append(bounds, bound{source: source, value: value})
		}
	}
	if len(bounds) == 0 {
		return Result{}
	}
	best := bounds[0]
	for _, candidate := range bounds[1:] {
		if candidate.value < best.value {
			best = candidate
		}
	}
	effective := best.value
	return Result{Effective: &effective, Source: best.source}
}

// Clone copies a token-limit pointer without sharing the backing value.
func Clone(value *int64) *int64 {
	if value == nil {
		return nil
	}
	copied := *value
	return &copied
}

// FromInt converts a positive int pointer into an int64 pointer.
func FromInt(value *int) *int64 {
	if value == nil || *value < 1 {
		return nil
	}
	converted := int64(*value)
	return &converted
}
