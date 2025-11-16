"use client"

import * as React from "react"
import { apiClient } from "@/lib/api-client"
import { ConfigSchema } from "@/types"
import { ConfigPanel } from "@/components/config/ConfigPanel"

export default function Home() {
  const [schema, setSchema] = React.useState<ConfigSchema | null>(null)
  const [config, setConfig] = React.useState<Record<string, any>>({})
  const [loading, setLoading] = React.useState(true)
  const [error, setError] = React.useState<string | null>(null)

  React.useEffect(() => {
    async function loadSchema() {
      try {
        setLoading(true)
        const schemaData = await apiClient.getConfigSchema()
        setSchema(schemaData)
        setConfig(schemaData.defaults)
        setError(null)
      } catch (err) {
        console.error("Failed to load configuration schema:", err)
        setError(
          err instanceof Error
            ? err.message
            : "Failed to connect to backend API. Please ensure the API server is running on http://localhost:8000"
        )
      } finally {
        setLoading(false)
      }
    }

    loadSchema()
  }, [])

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary mx-auto mb-4"></div>
          <p className="text-muted-foreground">Loading configuration schema...</p>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="min-h-screen flex items-center justify-center p-4">
        <div className="max-w-md w-full bg-destructive/10 border border-destructive rounded-lg p-6">
          <h2 className="text-lg font-semibold text-destructive mb-2">
            Failed to Load Configuration
          </h2>
          <p className="text-sm text-muted-foreground mb-4">{error}</p>
          <div className="text-xs text-muted-foreground">
            <p className="mb-2">To start the backend API:</p>
            <pre className="bg-muted p-2 rounded">
              cd /path/to/redo
              <br />
              python src/diaremot/web/server.py
            </pre>
          </div>
        </div>
      </div>
    )
  }

  if (!schema) {
    return null
  }

  return (
    <div className="min-h-screen bg-background">
      <header className="border-b">
        <div className="container mx-auto px-4 py-6">
          <h1 className="text-3xl font-bold">DiaRemot Configuration</h1>
          <p className="text-muted-foreground mt-1">
            Configure all {Object.keys(schema.parameters).length} pipeline parameters
          </p>
        </div>
      </header>

      <main className="container mx-auto px-4 py-8">
        <ConfigPanel schema={schema} values={config} onChange={setConfig} />

        {/* Config Preview (for debugging) */}
        <details className="mt-8">
          <summary className="cursor-pointer text-sm font-medium text-muted-foreground hover:text-foreground">
            View Current Configuration (JSON)
          </summary>
          <pre className="mt-4 p-4 bg-muted rounded-lg text-xs overflow-auto max-h-96">
            {JSON.stringify(config, null, 2)}
          </pre>
        </details>
      </main>
    </div>
  )
}
