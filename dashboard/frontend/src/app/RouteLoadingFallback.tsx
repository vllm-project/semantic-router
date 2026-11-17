import React from 'react'
import { PageContainer, Skeleton } from '../ui'

/** Skeleton of a page header and content while a route chunk loads. */
const RouteLoadingFallback: React.FC = () => (
  <PageContainer>
    <div className="mb-6 flex items-start gap-3.5" aria-busy="true" aria-label="Loading page">
      <Skeleton className="size-11 rounded-xl" />
      <div className="flex flex-1 flex-col gap-2 pt-1">
        <Skeleton className="h-5 w-48" />
        <Skeleton className="h-3.5 w-96 max-w-full" />
      </div>
    </div>
    <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
      {Array.from({ length: 4 }).map((_, index) => (
        <Skeleton key={index} className="h-24 rounded-xl" />
      ))}
    </div>
    <Skeleton className="mt-4 h-72 rounded-xl" />
  </PageContainer>
)

export default RouteLoadingFallback
