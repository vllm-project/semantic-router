package classification

// CategoryInitializer prepares a category backend before publication.
type CategoryInitializer interface {
	Init(modelID string, useCPU bool, numClasses ...int) error
}
