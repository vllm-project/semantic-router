export function transitionFromViewToEdit(onClose: () => void, onEdit?: () => void) {
  onClose()
  onEdit?.()
}
