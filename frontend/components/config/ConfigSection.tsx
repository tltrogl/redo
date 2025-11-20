"use client"

import * as React from "react"
import { ConfigParameter } from "@/types"
import { ParamControl } from "./ParamControl"
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"

interface ConfigSectionProps {
  title: string
  description: string
  parameters: ConfigParameter[]
  values: Record<string, any>
  onChange: (paramName: string, value: any) => void
  collapsible?: boolean
}

export function ConfigSection({
  title,
  description,
  parameters,
  values,
  onChange,
  collapsible = false,
}: ConfigSectionProps) {
  const [isOpen, setIsOpen] = React.useState(!collapsible)

  // Separate basic and advanced parameters
  const basicParams = parameters.filter((p) => !p.advanced)
  const advancedParams = parameters.filter((p) => p.advanced)

  return (
    <Card className="mb-4">
      <CardHeader>
        <div
          className={collapsible ? "cursor-pointer" : ""}
          onClick={() => collapsible && setIsOpen(!isOpen)}
        >
          <div className="flex items-center justify-between">
            <div className="flex-1">
              <CardTitle className="text-lg">{title}</CardTitle>
              <CardDescription>{description}</CardDescription>
            </div>
            {collapsible && (
              <span className="text-sm text-muted-foreground">
                {isOpen ? "▼" : "▶"}
              </span>
            )}
          </div>
        </div>
      </CardHeader>

      {isOpen && (
        <CardContent>
          <div className="space-y-1">
            {basicParams.map((param) => (
              <ParamControl
                key={param.name}
                param={param}
                value={values[param.name]}
                onChange={(value) => onChange(param.name, value)}
              />
            ))}

            {advancedParams.length > 0 && (
              <details className="mt-4">
                <summary className="cursor-pointer text-sm font-medium text-muted-foreground hover:text-foreground">
                  Advanced Settings ({advancedParams.length})
                </summary>
                <div className="mt-4 space-y-1 pl-4 border-l-2 border-muted">
                  {advancedParams.map((param) => (
                    <ParamControl
                      key={param.name}
                      param={param}
                      value={values[param.name]}
                      onChange={(value) => onChange(param.name, value)}
                    />
                  ))}
                </div>
              </details>
            )}
          </div>
        </CardContent>
      )}
    </Card>
  )
}
