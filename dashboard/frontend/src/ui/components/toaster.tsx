import { Toaster as Sonner, toast } from 'sonner'

export { toast }

export function Toaster({ theme }: { theme: 'light' | 'dark' }) {
  return (
    <Sonner
      theme={theme}
      position="bottom-right"
      gap={8}
      toastOptions={{
        classNames: {
          toast:
            'group !rounded-lg !border !border-border !bg-popover !text-popover-foreground !shadow-lg !font-sans !text-[13px]',
          description: '!text-muted-foreground',
          actionButton: '!bg-primary !text-primary-foreground',
          cancelButton: '!bg-secondary !text-secondary-foreground',
        },
      }}
    />
  )
}
