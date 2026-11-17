import * as React from 'react'
import { Tabs as TabsPrimitive } from 'radix-ui'
import { cn } from '../lib/cn'

type TabsVariant = 'line' | 'pill'
const TabsVariantContext = React.createContext<TabsVariant>('line')

export const Tabs = React.forwardRef<
  React.ElementRef<typeof TabsPrimitive.Root>,
  React.ComponentPropsWithoutRef<typeof TabsPrimitive.Root> & { variant?: TabsVariant }
>(({ className, variant = 'line', ...props }, ref) => (
  <TabsVariantContext.Provider value={variant}>
    <TabsPrimitive.Root
      ref={ref}
      data-slot="tabs"
      className={cn('flex flex-col gap-4', className)}
      {...props}
    />
  </TabsVariantContext.Provider>
))
Tabs.displayName = 'Tabs'

export const TabsList = React.forwardRef<
  React.ElementRef<typeof TabsPrimitive.List>,
  React.ComponentPropsWithoutRef<typeof TabsPrimitive.List>
>(({ className, ...props }, ref) => {
  const variant = React.useContext(TabsVariantContext)
  return (
    <TabsPrimitive.List
      ref={ref}
      data-slot="tabs-list"
      className={cn(
        variant === 'line'
          ? 'relative flex items-center gap-1 border-b border-border'
          : 'inline-flex w-fit items-center gap-0.5 rounded-lg border border-border bg-surface-subtle p-0.5',
        className,
      )}
      {...props}
    />
  )
})
TabsList.displayName = 'TabsList'

export const TabsTrigger = React.forwardRef<
  React.ElementRef<typeof TabsPrimitive.Trigger>,
  React.ComponentPropsWithoutRef<typeof TabsPrimitive.Trigger>
>(({ className, ...props }, ref) => {
  const variant = React.useContext(TabsVariantContext)
  return (
    <TabsPrimitive.Trigger
      ref={ref}
      data-slot="tabs-trigger"
      className={cn(
        "inline-flex items-center gap-1.5 whitespace-nowrap text-[13px] font-medium outline-none transition-colors focus-visible:ring-[3px] focus-visible:ring-ring/40 disabled:pointer-events-none disabled:opacity-50 [&_svg:not([class*='size-'])]:size-4",
        variant === 'line'
          ? 'relative -mb-px h-9 rounded-none border-b-2 border-transparent px-2.5 text-muted-foreground hover:text-foreground data-[state=active]:border-foreground data-[state=active]:text-foreground'
          : 'h-7 rounded-md px-2.5 text-muted-foreground hover:text-foreground data-[state=active]:bg-surface data-[state=active]:text-foreground data-[state=active]:shadow-sm dark:data-[state=active]:bg-accent',
        className,
      )}
      {...props}
    />
  )
})
TabsTrigger.displayName = 'TabsTrigger'

export const TabsContent = React.forwardRef<
  React.ElementRef<typeof TabsPrimitive.Content>,
  React.ComponentPropsWithoutRef<typeof TabsPrimitive.Content>
>(({ className, ...props }, ref) => (
  <TabsPrimitive.Content
    ref={ref}
    data-slot="tabs-content"
    className={cn('outline-none data-[state=active]:animate-fade-up', className)}
    {...props}
  />
))
TabsContent.displayName = 'TabsContent'
