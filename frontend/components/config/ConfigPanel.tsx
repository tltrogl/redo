"use client"

import * as React from "react"
import { ConfigSchema, ConfigParameter } from "@/types"
import { ConfigSection } from "./ConfigSection"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Button } from "@/components/ui/button"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { Label } from "@/components/ui/label"

interface ConfigPanelProps {
  schema: ConfigSchema
  values: Record<string, any>
  onChange: (values: Record<string, any>) => void
}

export function ConfigPanel({ schema, values, onChange }: ConfigPanelProps) {
  const [selectedPreset, setSelectedPreset] = React.useState<string | null>(null)

  // Group parameters by group
  const parametersByGroup = React.useMemo(() => {
    const groups: Record<string, ConfigParameter[]> = {}

    Object.entries(schema.parameters).forEach(([name, param]) => {
      const group = param.group || "other"
      if (!groups[group]) {
        groups[group] = []
      }
      groups[group].push({ ...param, name })
    })

    return groups
  }, [schema.parameters])

  // Sort groups by order
  const sortedGroups = React.useMemo(() => {
    return Object.entries(schema.groups)
      .sort(([, a], [, b]) => a.order - b.order)
      .map(([key]) => key)
  }, [schema.groups])

  const handleParamChange = (paramName: string, value: any) => {
    onChange({
      ...values,
      [paramName]: value,
    })
    // Clear preset selection when manual changes are made
    setSelectedPreset(null)
  }

  const handlePresetChange = (presetName: string) => {
    setSelectedPreset(presetName)
    const presetValues = schema.presets[presetName]
    if (presetValues) {
      onChange({
        ...schema.defaults,
        ...presetValues,
      })
    }
  }

  const handleReset = () => {
    onChange({ ...schema.defaults })
    setSelectedPreset(null)
  }

  return (
    <div className="w-full">
      {/* Preset Selector */}
      <div className="mb-6 p-4 bg-muted/50 rounded-lg">
        <div className="flex items-end gap-4">
          <div className="flex-1">
            <Label htmlFor="preset-select">Configuration Preset</Label>
            <Select value={selectedPreset ?? ""} onValueChange={handlePresetChange}>
              <SelectTrigger id="preset-select" className="mt-1">
                <SelectValue placeholder="Select a preset or customize manually" />
              </SelectTrigger>
              <SelectContent>
                {Object.keys(schema.presets).map((presetName) => (
                  <SelectItem key={presetName} value={presetName}>
                    {presetName.charAt(0).toUpperCase() + presetName.slice(1)}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
          <Button variant="outline" onClick={handleReset}>
            Reset to Defaults
          </Button>
        </div>
        {selectedPreset && (
          <p className="text-sm text-muted-foreground mt-2">
            Using <strong>{selectedPreset}</strong> preset. Make changes below to customize.
          </p>
        )}
      </div>

      {/* Parameter Tabs */}
      <Tabs defaultValue={sortedGroups[0]} className="w-full">
        <TabsList className="grid w-full grid-cols-6">
          {sortedGroups.map((groupKey) => {
            const group = schema.groups[groupKey]
            return (
              <TabsTrigger key={groupKey} value={groupKey} className="text-xs">
                <span className="mr-1">{group.icon}</span>
                <span className="hidden md:inline">{group.label}</span>
              </TabsTrigger>
            )
          })}
        </TabsList>

        {sortedGroups.map((groupKey) => {
          const group = schema.groups[groupKey]
          const groupParams = parametersByGroup[groupKey] || []

          return (
            <TabsContent key={groupKey} value={groupKey} className="mt-4">
              <ConfigSection
                title={group.label}
                description={group.description}
                parameters={groupParams}
                values={values}
                onChange={handleParamChange}
              />
            </TabsContent>
          )
        })}
      </Tabs>

      {/* Parameter Count Summary */}
      <div className="mt-6 p-4 bg-muted/30 rounded-lg">
        <p className="text-sm text-muted-foreground">
          <strong>{Object.keys(schema.parameters).length}</strong> total configurable
          parameters across <strong>{sortedGroups.length}</strong> categories
        </p>
      </div>
    </div>
  )
}
