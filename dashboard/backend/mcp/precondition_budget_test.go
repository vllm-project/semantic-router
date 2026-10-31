package mcp

import "time"

// preconditionBudget bounds the waits for a helper process or an
// initialization to start: the setup contexts around Connect, the "did not
// start" selects, and the stdio-helper marker polls. Those waits fail only
// when the step never happens, so on a healthy run they cost nothing and a
// longer budget hides no regression. The shutdown assertions - Close
// cancelling, joining and reaping - keep their own tight timers, which must
// stay below the code's fallback timeouts.
const preconditionBudget = 30 * time.Second
