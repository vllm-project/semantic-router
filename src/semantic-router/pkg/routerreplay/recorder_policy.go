package routerreplay

// CapturePolicy is the effective content policy for one request. Store capacity
// belongs to the shared recorder and is not a per-request setting.
type CapturePolicy struct {
	CaptureRequestBody     bool
	CaptureResponseBody    bool
	MaxBodyBytes           int
	MaxPromptBytes         int
	MaxToolTraceBytes      int
	MaxRawToolTraceBytes   int
	MaxToolTraceTotalBytes int
	MaxToolTraceSteps      int
}

// WithCapturePolicy returns an independent policy view of the same recorder.
// Requests can retain this view through response capture without changing the
// policy of concurrent requests. Store, outcome queue, lifecycle transitions,
// and close ownership remain shared; closing any view closes the owner once.
func (r *Recorder) WithCapturePolicy(policy CapturePolicy) *Recorder {
	if r == nil {
		return nil
	}
	base := r.policySnapshot()
	view := &Recorder{
		recorderState:          r.recorderState,
		maxPromptBytes:         base.maxPromptBytes,
		maxToolTraceBytes:      base.maxToolTraceBytes,
		maxRawToolTraceBytes:   base.maxRawToolTraceBytes,
		maxToolTraceTotalBytes: base.maxToolTraceTotalBytes,
		maxToolTraceSteps:      base.maxToolTraceSteps,
	}
	view.SetCapturePolicy(policy.CaptureRequestBody, policy.CaptureResponseBody, policy.MaxBodyBytes)
	view.SetMaxPromptBytes(policy.MaxPromptBytes)
	view.SetMaxToolTraceBytes(policy.MaxToolTraceBytes)
	view.SetMaxRawToolTraceBytes(policy.MaxRawToolTraceBytes)
	view.SetMaxToolTraceTotalBytes(policy.MaxToolTraceTotalBytes)
	view.SetMaxToolTraceSteps(policy.MaxToolTraceSteps)
	return view
}
