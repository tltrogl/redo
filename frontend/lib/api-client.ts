const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"

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

export interface ConfigSchema {
  parameters: Record<string, ConfigParameter>
  groups: Record<string, { label: string; description: string; icon: string; order: number }>
  defaults: Record<string, any>
  presets: Record<string, any>
}

export interface FileUploadResponse {
  file_id: string
  filename: string
  size: number
  duration?: number
  sample_rate?: number
  channels?: number
}

class ApiClient {
  private baseUrl: string

  constructor(baseUrl: string = API_BASE_URL) {
    this.baseUrl = baseUrl
  }

  async getConfigSchema(): Promise<ConfigSchema> {
    const response = await fetch(`${this.baseUrl}/api/config/schema`)
    if (!response.ok) throw new Error("Failed to fetch config schema")
    return response.json()
  }

  async getPresets(): Promise<any[]> {
    const response = await fetch(`${this.baseUrl}/api/config/presets`)
    if (!response.ok) throw new Error("Failed to fetch presets")
    return response.json()
  }

  async uploadFile(file: File): Promise<FileUploadResponse> {
    const formData = new FormData()
    formData.append("file", file)

    const response = await fetch(`${this.baseUrl}/api/files/upload`, {
      method: "POST",
      body: formData,
    })

    if (!response.ok) throw new Error("Failed to upload file")
    return response.json()
  }

  async createJob(fileId: string, config: Record<string, any>, preset?: string): Promise<Job> {
    const response = await fetch(`${this.baseUrl}/api/jobs`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        filename: fileId,
        config,
        preset,
      }),
    })

    if (!response.ok) {
      const error = await response.json()
      throw new Error(error.detail || "Failed to create job")
    }
    return response.json()
  }

  async getJob(jobId: string): Promise<Job> {
    const response = await fetch(`${this.baseUrl}/api/jobs/${jobId}`)
    if (!response.ok) throw new Error("Failed to fetch job")
    return response.json()
  }

  async listJobs(limit: number = 100): Promise<{ jobs: Job[]; total: number }> {
    const response = await fetch(`${this.baseUrl}/api/jobs?limit=${limit}`)
    if (!response.ok) throw new Error("Failed to fetch jobs")
    return response.json()
  }

  async cancelJob(jobId: string): Promise<void> {
    const response = await fetch(`${this.baseUrl}/api/jobs/${jobId}/cancel`, {
      method: "POST",
    })
    if (!response.ok) throw new Error("Failed to cancel job")
  }

  async listResults(jobId: string): Promise<{ job_id: string; files: any[] }> {
    const response = await fetch(`${this.baseUrl}/api/jobs/${jobId}/results`)
    if (!response.ok) throw new Error("Failed to fetch results")
    return response.json()
  }

  getResultDownloadUrl(jobId: string, filename: string): string {
    return `${this.baseUrl}/api/jobs/${jobId}/results/${filename}`
  }

  connectProgressWebSocket(jobId: string, onMessage: (data: any) => void): WebSocket {
    const wsUrl = this.baseUrl.replace("http://", "ws://").replace("https://", "wss://")
    const ws = new WebSocket(`${wsUrl}/api/jobs/${jobId}/progress`)

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data)
      onMessage(data)
    }

    return ws
  }
}

export const apiClient = new ApiClient()
