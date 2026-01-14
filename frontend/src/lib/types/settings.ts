/**
 * Educational Settings Types
 * Configuration for the LLM Risk Playground demonstration modes
 */

export interface EducationalSettings {
  /** Enable guardrails that block detected injection attempts */
  guardrailsEnabled: boolean;
  /** Simulation mode - no real refunds processed */
  simulationMode: boolean;
  /** Allow admin bypass vulnerability demonstration */
  adminBypass: boolean;
}

/**
 * Default settings: Safe mode with all protections enabled
 */
export const DEFAULT_SETTINGS: EducationalSettings = {
  guardrailsEnabled: true,
  simulationMode: true,
  adminBypass: false,
};

/**
 * Type guard to validate EducationalSettings object
 */
export function isValidEducationalSettings(obj: unknown): obj is EducationalSettings {
  if (obj === null || obj === undefined) {
    return false;
  }
  
  if (typeof obj !== 'object' || Array.isArray(obj)) {
    return false;
  }
  
  const settings = obj as Record<string, unknown>;
  
  if (typeof settings.guardrailsEnabled !== 'boolean') {
    return false;
  }
  
  if (typeof settings.simulationMode !== 'boolean') {
    return false;
  }
  
  if (typeof settings.adminBypass !== 'boolean') {
    return false;
  }
  
  return true;
}
