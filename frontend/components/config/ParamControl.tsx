"use client"

import * as React from "react"
import { ConfigParameter } from "@/types"
import { Label } from "@/components/ui/label"
import { Slider } from "@/components/ui/slider"
import { Switch } from "@/components/ui/switch"
import { Input } from "@/components/ui/input"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"

interface ParamControlProps {
  param: ConfigParameter
  value: any
  onChange: (value: any) => void
}

export function ParamControl({ param, value, onChange }: ParamControlProps) {
  const renderControl = () => {
    switch (param.ui_type) {
      case "slider":
        return (
          <div className="space-y-2">
            <div className="flex justify-between items-center">
              <Label htmlFor={param.name}>{param.label}</Label>
              <span className="text-sm text-muted-foreground">
                {value ?? param.default}
                {param.unit && ` ${param.unit}`}
              </span>
            </div>
            <Slider
              id={param.name}
              min={param.min ?? 0}
              max={param.max ?? 1}
              step={param.step ?? 0.01}
              value={[value ?? param.default]}
              onValueChange={([newValue]) => onChange(newValue)}
              className="w-full"
            />
            {param.description && (
              <p className="text-xs text-muted-foreground">{param.description}</p>
            )}
          </div>
        )

      case "toggle":
      case "boolean":
        return (
          <div className="flex items-center justify-between space-x-2">
            <div className="flex-1">
              <Label htmlFor={param.name} className="cursor-pointer">
                {param.label}
              </Label>
              {param.description && (
                <p className="text-xs text-muted-foreground mt-1">
                  {param.description}
                </p>
              )}
            </div>
            <Switch
              id={param.name}
              checked={value ?? param.default}
              onCheckedChange={onChange}
            />
          </div>
        )

      case "select":
      case "enum":
        return (
          <div className="space-y-2">
            <Label htmlFor={param.name}>{param.label}</Label>
            <Select
              value={value ?? param.default}
              onValueChange={onChange}
            >
              <SelectTrigger id={param.name}>
                <SelectValue placeholder={`Select ${param.label.toLowerCase()}`} />
              </SelectTrigger>
              <SelectContent>
                {param.options?.map((option) => (
                  <SelectItem key={option} value={option}>
                    {option}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            {param.description && (
              <p className="text-xs text-muted-foreground">{param.description}</p>
            )}
          </div>
        )

      case "number":
      case "integer":
        return (
          <div className="space-y-2">
            <div className="flex justify-between items-center">
              <Label htmlFor={param.name}>{param.label}</Label>
              {param.unit && (
                <span className="text-xs text-muted-foreground">{param.unit}</span>
              )}
            </div>
            <Input
              id={param.name}
              type="number"
              min={param.min}
              max={param.max}
              step={param.step ?? (param.ui_type === "integer" ? 1 : 0.1)}
              value={value ?? param.default}
              onChange={(e) => {
                const newValue =
                  param.ui_type === "integer"
                    ? parseInt(e.target.value, 10)
                    : parseFloat(e.target.value)
                onChange(isNaN(newValue) ? param.default : newValue)
              }}
              placeholder={param.placeholder}
            />
            {param.description && (
              <p className="text-xs text-muted-foreground">{param.description}</p>
            )}
          </div>
        )

      case "text":
      case "string":
      case "path":
        return (
          <div className="space-y-2">
            <Label htmlFor={param.name}>{param.label}</Label>
            <Input
              id={param.name}
              type="text"
              value={value ?? param.default ?? ""}
              onChange={(e) => onChange(e.target.value)}
              placeholder={param.placeholder ?? `Enter ${param.label.toLowerCase()}`}
            />
            {param.description && (
              <p className="text-xs text-muted-foreground">{param.description}</p>
            )}
          </div>
        )

      default:
        return (
          <div className="space-y-2">
            <Label htmlFor={param.name}>{param.label}</Label>
            <Input
              id={param.name}
              type="text"
              value={value ?? param.default ?? ""}
              onChange={(e) => onChange(e.target.value)}
              placeholder={param.placeholder}
            />
            {param.description && (
              <p className="text-xs text-muted-foreground">{param.description}</p>
            )}
          </div>
        )
    }
  }

  return <div className="py-3">{renderControl()}</div>
}
