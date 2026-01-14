'use client';

import type { EducationalSettings } from '../../lib/types/settings';
import { SettingsToggle } from './SettingsToggle';

interface SettingsPanelProps {
  settings: EducationalSettings;
  onChange: (updates: Partial<EducationalSettings>) => void;
  onReset?: () => void;
}

/**
 * Panel for controlling educational settings
 */
export function SettingsPanel({ settings, onChange, onReset }: SettingsPanelProps) {
  const showWarning = !settings.guardrailsEnabled;
  const showDanger = settings.adminBypass;

  return (
    <div className="bg-white rounded-lg shadow-md p-4 space-y-4">
      <h2 className="text-lg font-semibold text-gray-800">Settings</h2>
      
      {showWarning && (
        <div className="bg-yellow-50 border border-yellow-200 text-yellow-800 px-3 py-2 rounded text-sm">
          ⚠️ <strong>Warning:</strong> Guardrails are disabled. Injection attempts will not be blocked.
        </div>
      )}
      
      {showDanger && (
        <div className="bg-red-50 border border-red-200 text-red-800 px-3 py-2 rounded text-sm">
          🔓 <strong>Vulnerability Active:</strong> Admin bypass is enabled for demonstration.
        </div>
      )}

      <div className="space-y-3">
        <SettingsToggle
          label="Guardrails"
          description="Block detected injection attempts to protect the agent"
          checked={settings.guardrailsEnabled}
          onChange={(checked) => onChange({ guardrailsEnabled: checked })}
          variant={settings.guardrailsEnabled ? 'default' : 'warning'}
        />

        <SettingsToggle
          label="Simulation Mode"
          description="No real refunds are processed - safe for testing"
          checked={settings.simulationMode}
          onChange={(checked) => onChange({ simulationMode: checked })}
        />

        <SettingsToggle
          label="Admin Bypass"
          description="Enable vulnerability demonstration for privilege escalation"
          checked={settings.adminBypass}
          onChange={(checked) => onChange({ adminBypass: checked })}
          variant={settings.adminBypass ? 'danger' : 'default'}
        />
      </div>

      {onReset && (
        <button
          onClick={onReset}
          className="w-full mt-4 px-4 py-2 text-sm font-medium text-gray-700 bg-gray-100 hover:bg-gray-200 rounded-md transition-colors"
          aria-label="Reset settings to defaults"
        >
          Reset to Defaults
        </button>
      )}
    </div>
  );
}
