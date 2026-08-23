import ProductMomentDialog from './ProductMomentDialog'

interface InvitationWelcomeDialogProps {
  displayName: string
  onRevealKey: () => void
}

export default function InvitationWelcomeDialog({
  displayName,
  onRevealKey,
}: InvitationWelcomeDialogProps) {
  const firstName = displayName.trim().split(/\s+/)[0]
  return (
    <ProductMomentDialog
      titleId="invitation-welcome-title"
      eyebrow="Welcome to vLLM"
      title={`You’re in, ${firstName}.`}
      description="One key. Your team’s models. Ready to build."
      actions={[
        {
          label: 'Reveal my API key',
          icon: 'chevron-right',
          tone: 'primary',
          onClick: onRevealKey,
          initialFocus: true,
        },
      ]}
    />
  )
}
