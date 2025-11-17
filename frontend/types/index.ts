export interface ConfigParameter {
  name: string
  type: string
  default: any
  label: string
  description: string
  group: string
  ui_type: string
  advanced: boolean
  min?: number
  max?: number
  step?: number
  options?: string[]
  unit?: string
  nullable?: boolean
  placeholder?: string
}

export interface ConfigGroup {
  label: string
  description: string
  icon: string
  order: number
}

export interface ConfigSchema {
  parameters: Record<string, ConfigParameter>
  groups: Record<string, ConfigGroup>
  defaults: Record<string, any>
  presets: Record<string, any>
}

export interface JobProgress {
  stage: string | null
  stage_index: number
  stage_progress: number
  overall_progress: number
  message: string
}

export interface Job {
  job_id: string
  status: "pending" | "running" | "completed" | "failed" | "cancelled"
  filename: string
  created_at: string
  started_at: string | null
  completed_at: string | null
  progress: JobProgress
  error: string | null
  config: Record<string, any>
  results: Record<string, any> | null
}

export interface FileUploadInfo {
  file_id: string
  filename: string
  size: number
  duration?: number
  sample_rate?: number
  channels?: number
}
