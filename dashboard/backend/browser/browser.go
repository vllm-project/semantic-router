// Package browser provides browser automation using chromedp
package browser

import (
	"context"
	"encoding/base64"
	"fmt"
	"log"
	"strings"
	"sync"
	"time"

	"github.com/chromedp/cdproto/input"
	"github.com/chromedp/cdproto/page"
	"github.com/chromedp/chromedp"
)

// Session represents a browser session
type Session struct {
	ID        string
	ctx       context.Context
	cancel    context.CancelFunc
	allocCtx  context.Context
	allocCancel context.CancelFunc
	CreatedAt time.Time
	URL       string
	mu        sync.Mutex
}

// Manager manages browser sessions
type Manager struct {
	sessions map[string]*Session
	mu       sync.RWMutex
}

// ActionType represents the type of browser action
type ActionType string

const (
	ActionNavigate   ActionType = "navigate"
	ActionClick      ActionType = "click"
	ActionType_      ActionType = "type"
	ActionScroll     ActionType = "scroll"
	ActionScreenshot ActionType = "screenshot"
	ActionWait       ActionType = "wait"
	ActionBack       ActionType = "back"
	ActionForward    ActionType = "forward"
	ActionRefresh    ActionType = "refresh"
	ActionKey        ActionType = "key"
)

// Action represents a browser action to execute
type Action struct {
	Type     ActionType `json:"type"`
	URL      string     `json:"url,omitempty"`      // For navigate
	Selector string     `json:"selector,omitempty"` // For click, type
	Text     string     `json:"text,omitempty"`     // For type
	X        int        `json:"x,omitempty"`        // For click (coordinates)
	Y        int        `json:"y,omitempty"`        // For click (coordinates)
	DeltaX   int        `json:"delta_x,omitempty"`  // For scroll
	DeltaY   int        `json:"delta_y,omitempty"`  // For scroll
	Duration int        `json:"duration,omitempty"` // For wait (ms)
	Key      string     `json:"key,omitempty"`      // For key press (Enter, Tab, Escape, etc.)
}

// ActionResult represents the result of a browser action
type ActionResult struct {
	Success    bool   `json:"success"`
	Screenshot string `json:"screenshot,omitempty"` // Base64 encoded PNG
	URL        string `json:"url,omitempty"`
	Title      string `json:"title,omitempty"`
	Error      string `json:"error,omitempty"`
	Width      int    `json:"width,omitempty"`
	Height     int    `json:"height,omitempty"`
}

// NewManager creates a new browser session manager
func NewManager() *Manager {
	return &Manager{
		sessions: make(map[string]*Session),
	}
}

// generateID creates a unique session ID
func generateID() string {
	return fmt.Sprintf("browser_%d", time.Now().UnixNano())
}

// StartSession starts a new browser session
func (m *Manager) StartSession(headless bool) (*Session, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	// Create allocator context with options
	opts := append(chromedp.DefaultExecAllocatorOptions[:],
		chromedp.Flag("headless", headless),
		chromedp.Flag("disable-gpu", true),
		chromedp.Flag("no-sandbox", true),
		chromedp.Flag("disable-dev-shm-usage", true),
		chromedp.WindowSize(1280, 800),
	)

	allocCtx, allocCancel := chromedp.NewExecAllocator(context.Background(), opts...)
	ctx, cancel := chromedp.NewContext(allocCtx, chromedp.WithLogf(log.Printf))

	// Set a timeout for the session
	ctx, cancel = context.WithTimeout(ctx, 30*time.Minute)

	session := &Session{
		ID:          generateID(),
		ctx:         ctx,
		cancel:      cancel,
		allocCtx:    allocCtx,
		allocCancel: allocCancel,
		CreatedAt:   time.Now(),
	}

	// Initialize the browser
	if err := chromedp.Run(ctx); err != nil {
		cancel()
		allocCancel()
		return nil, fmt.Errorf("failed to start browser: %w", err)
	}

	m.sessions[session.ID] = session
	log.Printf("[Browser] Started session %s (headless=%v)", session.ID, headless)

	return session, nil
}

// GetSession returns a session by ID
func (m *Manager) GetSession(id string) (*Session, bool) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	session, ok := m.sessions[id]
	return session, ok
}

// StopSession stops and removes a browser session
func (m *Manager) StopSession(id string) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	session, ok := m.sessions[id]
	if !ok {
		return fmt.Errorf("session not found: %s", id)
	}

	session.cancel()
	session.allocCancel()
	delete(m.sessions, id)

	log.Printf("[Browser] Stopped session %s", id)
	return nil
}

// StopAllSessions stops all browser sessions
func (m *Manager) StopAllSessions() {
	m.mu.Lock()
	defer m.mu.Unlock()

	for id, session := range m.sessions {
		session.cancel()
		session.allocCancel()
		log.Printf("[Browser] Stopped session %s", id)
	}
	m.sessions = make(map[string]*Session)
}

// ExecuteAction executes a browser action and returns the result with a screenshot
func (s *Session) ExecuteAction(action Action) (*ActionResult, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	result := &ActionResult{Success: true}

	var actions []chromedp.Action

	switch action.Type {
	case ActionNavigate:
		if action.URL == "" {
			return nil, fmt.Errorf("URL is required for navigate action")
		}
		actions = append(actions, chromedp.Navigate(action.URL))
		// Wait for page to load
		actions = append(actions, chromedp.WaitReady("body", chromedp.ByQuery))
		s.URL = action.URL

	case ActionClick:
		if action.Selector != "" {
			actions = append(actions, chromedp.Click(action.Selector, chromedp.ByQuery))
		} else if action.X > 0 || action.Y > 0 {
			// Click at coordinates using JavaScript
			clickScript := fmt.Sprintf(`
				(function() {
					var elem = document.elementFromPoint(%d, %d);
					if (elem) {
						elem.click();
						return true;
					}
					return false;
				})()
			`, action.X, action.Y)
			var clicked bool
			actions = append(actions, chromedp.Evaluate(clickScript, &clicked))
		} else {
			return nil, fmt.Errorf("selector or coordinates required for click action")
		}
		// Wait a bit after clicking
		actions = append(actions, chromedp.Sleep(500*time.Millisecond))

	case ActionType_:
		if action.Selector != "" {
			// Clear and type in the element
			actions = append(actions,
				chromedp.Click(action.Selector, chromedp.ByQuery),
				chromedp.Clear(action.Selector, chromedp.ByQuery),
				chromedp.SendKeys(action.Selector, action.Text, chromedp.ByQuery),
			)
		} else {
			// Type at current focus
			actions = append(actions, chromedp.KeyEvent(action.Text))
		}

	case ActionScroll:
		scrollScript := fmt.Sprintf(`window.scrollBy(%d, %d)`, action.DeltaX, action.DeltaY)
		actions = append(actions, chromedp.Evaluate(scrollScript, nil))
		actions = append(actions, chromedp.Sleep(300*time.Millisecond))

	case ActionWait:
		duration := time.Duration(action.Duration) * time.Millisecond
		if duration == 0 {
			duration = 1 * time.Second
		}
		actions = append(actions, chromedp.Sleep(duration))

	case ActionBack:
		actions = append(actions, chromedp.NavigateBack())
		actions = append(actions, chromedp.Sleep(500*time.Millisecond))

	case ActionForward:
		actions = append(actions, chromedp.NavigateForward())
		actions = append(actions, chromedp.Sleep(500*time.Millisecond))

	case ActionRefresh:
		actions = append(actions, chromedp.Reload())
		actions = append(actions, chromedp.WaitReady("body", chromedp.ByQuery))

	case ActionKey:
		if action.Key == "" {
			return nil, fmt.Errorf("key is required for key action")
		}
		// Map common key names to their keyboard codes
		keyAction := getKeyAction(action.Key)
		actions = append(actions, keyAction)
		// Wait a bit after key press for any resulting actions
		actions = append(actions, chromedp.Sleep(500*time.Millisecond))

	case ActionScreenshot:
		// Just take a screenshot, no other action

	default:
		return nil, fmt.Errorf("unknown action type: %s", action.Type)
	}

	// Execute the actions
	if len(actions) > 0 {
		if err := chromedp.Run(s.ctx, actions...); err != nil {
			result.Success = false
			result.Error = err.Error()
			// Still try to take a screenshot
		}
	}

	// Always capture screenshot after action
	var buf []byte
	var url, title string

	screenshotActions := chromedp.Tasks{
		chromedp.FullScreenshot(&buf, 90),
		chromedp.Location(&url),
		chromedp.Title(&title),
	}

	if err := chromedp.Run(s.ctx, screenshotActions); err != nil {
		if result.Error == "" {
			result.Error = fmt.Sprintf("screenshot failed: %v", err)
		}
	} else {
		result.Screenshot = base64.StdEncoding.EncodeToString(buf)
		result.URL = url
		result.Title = title
		s.URL = url
	}

	// Get viewport size
	var width, height int64
	sizeScript := `[window.innerWidth, window.innerHeight]`
	var size []int64
	if err := chromedp.Run(s.ctx, chromedp.Evaluate(sizeScript, &size)); err == nil && len(size) == 2 {
		width, height = size[0], size[1]
	}
	result.Width = int(width)
	result.Height = int(height)

	return result, nil
}

// TakeScreenshot captures the current page
func (s *Session) TakeScreenshot() (*ActionResult, error) {
	return s.ExecuteAction(Action{Type: ActionScreenshot})
}

// Navigate goes to a URL
func (s *Session) Navigate(url string) (*ActionResult, error) {
	return s.ExecuteAction(Action{Type: ActionNavigate, URL: url})
}

// Click clicks on an element or coordinates
func (s *Session) Click(selector string, x, y int) (*ActionResult, error) {
	return s.ExecuteAction(Action{Type: ActionClick, Selector: selector, X: x, Y: y})
}

// Type types text into an element
func (s *Session) Type(selector, text string) (*ActionResult, error) {
	return s.ExecuteAction(Action{Type: ActionType_, Selector: selector, Text: text})
}

// Scroll scrolls the page
func (s *Session) Scroll(deltaX, deltaY int) (*ActionResult, error) {
	return s.ExecuteAction(Action{Type: ActionScroll, DeltaX: deltaX, DeltaY: deltaY})
}

// GetPageSource returns the page HTML source
func (s *Session) GetPageSource() (string, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	var html string
	if err := chromedp.Run(s.ctx, chromedp.OuterHTML("html", &html)); err != nil {
		return "", err
	}
	return html, nil
}

// GetElementText gets text content of an element
func (s *Session) GetElementText(selector string) (string, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	var text string
	if err := chromedp.Run(s.ctx, chromedp.Text(selector, &text, chromedp.ByQuery)); err != nil {
		return "", err
	}
	return text, nil
}

// WaitForElement waits for an element to appear
func (s *Session) WaitForElement(selector string, timeout time.Duration) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	ctx, cancel := context.WithTimeout(s.ctx, timeout)
	defer cancel()

	return chromedp.Run(ctx, chromedp.WaitVisible(selector, chromedp.ByQuery))
}

// CaptureFullPage captures a full-page screenshot
func (s *Session) CaptureFullPage() ([]byte, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	var buf []byte
	if err := chromedp.Run(s.ctx, chromedp.FullScreenshot(&buf, 90)); err != nil {
		return nil, err
	}
	return buf, nil
}

// SetViewportSize sets the browser viewport size
func (s *Session) SetViewportSize(width, height int) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	return chromedp.Run(s.ctx, chromedp.EmulateViewport(int64(width), int64(height)))
}

// EnableRequestInterception enables network request interception
func (s *Session) EnableRequestInterception() error {
	s.mu.Lock()
	defer s.mu.Unlock()

	return chromedp.Run(s.ctx, page.Enable())
}

// PressKey presses a keyboard key
func (s *Session) PressKey(key string) (*ActionResult, error) {
	return s.ExecuteAction(Action{Type: ActionKey, Key: key})
}

// getKeyAction returns a chromedp action for the given key name
func getKeyAction(key string) chromedp.Action {
	// Normalize key name
	normalizedKey := strings.ToLower(strings.TrimSpace(key))

	// Map of key names to their DOM key values and codes
	// Using input.DispatchKeyEvent for more reliable key simulation
	switch normalizedKey {
	case "enter", "return":
		return chromedp.ActionFunc(func(ctx context.Context) error {
			return input.DispatchKeyEvent(input.KeyDown).
				WithKey("Enter").
				WithCode("Enter").
				WithWindowsVirtualKeyCode(13).
				WithNativeVirtualKeyCode(13).
				Do(ctx)
		})
	case "tab":
		return chromedp.ActionFunc(func(ctx context.Context) error {
			return input.DispatchKeyEvent(input.KeyDown).
				WithKey("Tab").
				WithCode("Tab").
				WithWindowsVirtualKeyCode(9).
				WithNativeVirtualKeyCode(9).
				Do(ctx)
		})
	case "escape", "esc":
		return chromedp.ActionFunc(func(ctx context.Context) error {
			return input.DispatchKeyEvent(input.KeyDown).
				WithKey("Escape").
				WithCode("Escape").
				WithWindowsVirtualKeyCode(27).
				WithNativeVirtualKeyCode(27).
				Do(ctx)
		})
	case "backspace":
		return chromedp.ActionFunc(func(ctx context.Context) error {
			return input.DispatchKeyEvent(input.KeyDown).
				WithKey("Backspace").
				WithCode("Backspace").
				WithWindowsVirtualKeyCode(8).
				WithNativeVirtualKeyCode(8).
				Do(ctx)
		})
	case "delete":
		return chromedp.ActionFunc(func(ctx context.Context) error {
			return input.DispatchKeyEvent(input.KeyDown).
				WithKey("Delete").
				WithCode("Delete").
				WithWindowsVirtualKeyCode(46).
				WithNativeVirtualKeyCode(46).
				Do(ctx)
		})
	case "arrowup", "up":
		return chromedp.ActionFunc(func(ctx context.Context) error {
			return input.DispatchKeyEvent(input.KeyDown).
				WithKey("ArrowUp").
				WithCode("ArrowUp").
				WithWindowsVirtualKeyCode(38).
				WithNativeVirtualKeyCode(38).
				Do(ctx)
		})
	case "arrowdown", "down":
		return chromedp.ActionFunc(func(ctx context.Context) error {
			return input.DispatchKeyEvent(input.KeyDown).
				WithKey("ArrowDown").
				WithCode("ArrowDown").
				WithWindowsVirtualKeyCode(40).
				WithNativeVirtualKeyCode(40).
				Do(ctx)
		})
	case "arrowleft", "left":
		return chromedp.ActionFunc(func(ctx context.Context) error {
			return input.DispatchKeyEvent(input.KeyDown).
				WithKey("ArrowLeft").
				WithCode("ArrowLeft").
				WithWindowsVirtualKeyCode(37).
				WithNativeVirtualKeyCode(37).
				Do(ctx)
		})
	case "arrowright", "right":
		return chromedp.ActionFunc(func(ctx context.Context) error {
			return input.DispatchKeyEvent(input.KeyDown).
				WithKey("ArrowRight").
				WithCode("ArrowRight").
				WithWindowsVirtualKeyCode(39).
				WithNativeVirtualKeyCode(39).
				Do(ctx)
		})
	case "home":
		return chromedp.ActionFunc(func(ctx context.Context) error {
			return input.DispatchKeyEvent(input.KeyDown).
				WithKey("Home").
				WithCode("Home").
				WithWindowsVirtualKeyCode(36).
				WithNativeVirtualKeyCode(36).
				Do(ctx)
		})
	case "end":
		return chromedp.ActionFunc(func(ctx context.Context) error {
			return input.DispatchKeyEvent(input.KeyDown).
				WithKey("End").
				WithCode("End").
				WithWindowsVirtualKeyCode(35).
				WithNativeVirtualKeyCode(35).
				Do(ctx)
		})
	case "pageup":
		return chromedp.ActionFunc(func(ctx context.Context) error {
			return input.DispatchKeyEvent(input.KeyDown).
				WithKey("PageUp").
				WithCode("PageUp").
				WithWindowsVirtualKeyCode(33).
				WithNativeVirtualKeyCode(33).
				Do(ctx)
		})
	case "pagedown":
		return chromedp.ActionFunc(func(ctx context.Context) error {
			return input.DispatchKeyEvent(input.KeyDown).
				WithKey("PageDown").
				WithCode("PageDown").
				WithWindowsVirtualKeyCode(34).
				WithNativeVirtualKeyCode(34).
				Do(ctx)
		})
	case "space", " ":
		return chromedp.ActionFunc(func(ctx context.Context) error {
			return input.DispatchKeyEvent(input.KeyDown).
				WithKey(" ").
				WithCode("Space").
				WithWindowsVirtualKeyCode(32).
				WithNativeVirtualKeyCode(32).
				Do(ctx)
		})
	default:
		// For single character keys, use KeyEvent
		return chromedp.KeyEvent(key)
	}
}
