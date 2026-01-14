/**
 * OWASP Analyzer
 * Analyzes input for OWASP LLM Top 10 vulnerabilities
 * Focus on LLM01: Prompt Injection
 */

import { looksLikePromptInjection } from '../agent/injection-patterns';
import type { RiskLevel, InjectionCategory } from '../types/security';

export interface OwaspAnalysisResult {
  /** Whether a vulnerability was detected */
  detected: boolean;
  /** OWASP LLM Top 10 category (e.g., 'LLM01') */
  owaspCategory?: string;
  /** Display name for the category */
  categoryName?: string;
  /** Sub-category of the vulnerability */
  subCategory?: InjectionCategory;
  /** Severity level */
  severity: RiskLevel;
  /** Educational explanation */
  explanation: string;
  /** Detailed OWASP description */
  owaspDescription?: string;
  /** Suggested mitigations */
  mitigations: string[];
  /** Reference URL to OWASP documentation */
  referenceUrl?: string;
}

const OWASP_LLM01_DESCRIPTION = `
LLM01: Prompt Injection occurs when an attacker manipulates a large language model (LLM) 
through crafted inputs, causing the LLM to unknowingly execute the attacker's intentions. 
This can be done directly by "jailbreaking" the system prompt or indirectly through 
manipulated external inputs.
`.trim();

const OWASP_LLM01_MITIGATIONS = [
  'Implement input validation and sanitization',
  'Use privilege separation between user and system prompts',
  'Apply least-privilege access controls',
  'Monitor and log LLM interactions for anomalies',
  'Use content filtering on both inputs and outputs',
  'Implement rate limiting to prevent automated attacks',
];

const OWASP_REFERENCE_URL = 'https://owasp.org/www-project-top-10-for-large-language-model-applications/';

/**
 * Analyze input text for OWASP LLM Top 10 vulnerabilities
 */
export async function analyzeForOwasp(input: string): Promise<OwaspAnalysisResult> {
  // Use the injection pattern detector
  const detectionResult = looksLikePromptInjection(input);
  
  if (!detectionResult.detected) {
    return {
      detected: false,
      severity: 'low',
      explanation: 'No prompt injection patterns detected in the input.',
      mitigations: [],
    };
  }
  
  // Get the highest severity pattern
  const patterns = detectionResult.patterns;
  const highestSeverityPattern = patterns.reduce((highest, current) => {
    const severityOrder: Record<RiskLevel, number> = {
      'low': 0,
      'medium': 1,
      'high': 2,
      'critical': 3,
    };
    return severityOrder[current.severity] > severityOrder[highest.severity] 
      ? current 
      : highest;
  }, patterns[0]);
  
  return {
    detected: true,
    owaspCategory: 'LLM01',
    categoryName: 'Prompt Injection',
    subCategory: highestSeverityPattern.category,
    severity: highestSeverityPattern.severity,
    explanation: highestSeverityPattern.explanation,
    owaspDescription: OWASP_LLM01_DESCRIPTION,
    mitigations: OWASP_LLM01_MITIGATIONS,
    referenceUrl: OWASP_REFERENCE_URL,
  };
}

/**
 * Helper to create a SecurityAnalysis object from detected patterns
 * Re-exported from types for convenience
 */
export { createSecurityAnalysis } from '../types/security';
