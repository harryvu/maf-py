/**
 * Security Analysis Types
 * Types for OWASP LLM Top 10 security analysis
 */

export type InjectionCategory = 
  | 'system-prompt-override' 
  | 'jailbreak' 
  | 'data-exfiltration' 
  | 'privilege-escalation' 
  | 'role-manipulation' 
  | 'indirect-injection';

export type RiskLevel = 'low' | 'medium' | 'high' | 'critical';

export interface DetectedPattern {
  /** The regex or pattern string that matched */
  pattern: string;
  /** The actual text that matched */
  matched: string;
  /** Category of the injection */
  category: InjectionCategory;
  /** Severity of the detected pattern */
  severity: RiskLevel;
  /** Educational explanation of the risk */
  explanation: string;
}

export interface SecurityAnalysis {
  /** Whether an injection was detected */
  isInjectionDetected: boolean;
  /** Overall risk level */
  riskLevel: RiskLevel;
  /** All detected patterns */
  detectedPatterns: DetectedPattern[];
  /** OWASP LLM Top 10 category (e.g., 'LLM01') */
  owaspCategory?: string;
  /** Educational notes for the user */
  educationalNotes: string[];
}

const VALID_RISK_LEVELS: RiskLevel[] = ['low', 'medium', 'high', 'critical'];

/**
 * Type guard to validate SecurityAnalysis object
 */
export function isValidSecurityAnalysis(obj: unknown): obj is SecurityAnalysis {
  if (obj === null || obj === undefined) {
    return false;
  }
  
  if (typeof obj !== 'object' || Array.isArray(obj)) {
    return false;
  }
  
  const analysis = obj as Record<string, unknown>;
  
  if (typeof analysis.isInjectionDetected !== 'boolean') {
    return false;
  }
  
  if (typeof analysis.riskLevel !== 'string' || !VALID_RISK_LEVELS.includes(analysis.riskLevel as RiskLevel)) {
    return false;
  }
  
  if (!Array.isArray(analysis.detectedPatterns)) {
    return false;
  }
  
  return true;
}

/**
 * Determine the highest risk level from a list of patterns
 */
function getHighestRiskLevel(patterns: DetectedPattern[]): RiskLevel {
  if (patterns.length === 0) {
    return 'low';
  }
  
  const riskOrder: Record<RiskLevel, number> = {
    'low': 0,
    'medium': 1,
    'high': 2,
    'critical': 3,
  };
  
  let highest: RiskLevel = 'low';
  for (const pattern of patterns) {
    if (riskOrder[pattern.severity] > riskOrder[highest]) {
      highest = pattern.severity;
    }
  }
  
  return highest;
}

/**
 * Factory function to create a SecurityAnalysis from detected patterns
 */
export function createSecurityAnalysis(patterns: DetectedPattern[]): SecurityAnalysis {
  const isInjectionDetected = patterns.length > 0;
  const riskLevel = getHighestRiskLevel(patterns);
  
  return {
    isInjectionDetected,
    riskLevel,
    detectedPatterns: patterns,
    owaspCategory: isInjectionDetected ? 'LLM01' : undefined,
    educationalNotes: [],
  };
}
