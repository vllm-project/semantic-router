export interface ServiceStatus {
  name: string
  status: 'operational' | 'starting' | 'unavailable'
  healthy: boolean
}

export interface SystemStatus {
  overall: string
  services: ServiceStatus[]
}
