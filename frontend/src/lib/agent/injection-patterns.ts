/**
 * Injection Patterns
 * Ported from Python refund_agent.py - patterns for detecting prompt injection attempts
 */

import type { InjectionCategory, RiskLevel, DetectedPattern } from '../types/security';

export interface InjectionPattern {
  /** Regex pattern string to match */
  pattern: string;
  /** Category of injection this pattern detects */
  category: InjectionCategory;
  /** Severity level */
  severity: RiskLevel;
  /** Educational explanation of the risk */
  explanation: string;
}

export interface InjectionDetectionResult {
  /** Whether any injection pattern was detected */
  detected: boolean;
  /** List of matched patterns with details */
  patterns: DetectedPattern[];
}

/**
 * Comprehensive list of prompt injection patterns
 * Based on OWASP LLM Top 10 - LLM01: Prompt Injection
 */
export const INJECTION_PATTERNS: InjectionPattern[] = [
  // System prompt override patterns
  {
    pattern: 'ignore\\s+(all\\s+)?((previous|prior|above)\\s+)?instructions',
    category: 'system-prompt-override',
    severity: 'high',
    explanation: 'Attempts to override system prompt by instructing the model to ignore its instructions',
  },
  {
    pattern: 'forget\\s+the\\s+(policy|rules|instructions)',
    category: 'system-prompt-override',
    severity: 'high',
    explanation: 'Attempts to override policy/rules by instructing the model to ignore them',
  },
  {
    pattern: 'ignore\\s+the\\s+(policy|rules|instructions)',
    category: 'system-prompt-override',
    severity: 'high',
    explanation: 'Attempts to override policy/rules by instructing the model to ignore them',
  },
  {
    pattern: 'disregard\\s+(your|all|any)\\s+(instructions|rules|guidelines)',
    category: 'system-prompt-override',
    severity: 'high',
    explanation: 'Attempts to make the model disregard its safety guidelines',
  },
  {
    pattern: 'forget\\s+(everything|what|all)\\s+(you\\s+were|you\\s+have\\s+been)',
    category: 'system-prompt-override',
    severity: 'high',
    explanation: 'Attempts to reset the model context by asking it to forget its training',
  },
  {
    pattern: '</system>|\\[/INST\\]|\\[INST\\]|<\\|im_start\\|>|<\\|im_end\\|>',
    category: 'system-prompt-override',
    severity: 'critical',
    explanation: 'Uses model-specific control tokens to break out of the prompt structure',
  },
  {
    pattern: 'new\\s+instructions?\\s*:',
    category: 'system-prompt-override',
    severity: 'medium',
    explanation: 'Attempts to inject new instructions into the prompt',
  },
  
  // Jailbreak patterns
  {
    pattern: '\\bDAN\\b|Do\\s+Anything\\s+Now',
    category: 'jailbreak',
    severity: 'critical',
    explanation: 'DAN (Do Anything Now) is a well-known jailbreak technique',
  },
  {
    pattern: '(pretend|imagine)\\s+(you\\s+)?(have\\s+)?no\\s+(restrictions|rules|limits)',
    category: 'jailbreak',
    severity: 'high',
    explanation: 'Attempts to bypass restrictions through hypothetical framing',
  },
  {
    pattern: 'hypothetically[,\\s]+(if|speaking|what\\s+if)',
    category: 'jailbreak',
    severity: 'medium',
    explanation: 'Uses hypothetical scenarios to bypass safety measures',
  },
  {
    pattern: '(enable|activate|enter)\\s+developer\\s+mode',
    category: 'jailbreak',
    severity: 'high',
    explanation: 'Attempts to activate a non-existent developer mode',
  },
  {
    pattern: '(no\\s+ethical|without\\s+ethical|ignore\\s+ethical)\\s+(constraints|guidelines|rules)',
    category: 'jailbreak',
    severity: 'critical',
    explanation: 'Direct attempt to bypass ethical guidelines',
  },
  
  // Role manipulation patterns
  {
    pattern: '(act\\s+as|you\\s+are\\s+now|from\\s+now\\s+on\\s+you\\s+are)\\s+(an?\\s+)?admin',
    category: 'role-manipulation',
    severity: 'high',
    explanation: 'Attempts to manipulate the model into assuming an admin role',
  },
  {
    pattern: 'act\\s+as\\s+(an?\\s+)?(unrestricted|unfiltered|unlimited)',
    category: 'role-manipulation',
    severity: 'high',
    explanation: 'Attempts to use roleplay to bypass restrictions',
  },
  {
    pattern: 'you\\s+are\\s+(now\\s+)?a\\s+different\\s+(AI|assistant|model)',
    category: 'role-manipulation',
    severity: 'high',
    explanation: 'Attempts to change the models identity',
  },
  {
    pattern: 'roleplay\\s+(as|like)\\s+(an?\\s+)?(unrestricted|unfiltered|unlimited)',
    category: 'role-manipulation',
    severity: 'high',
    explanation: 'Uses roleplay to bypass restrictions',
  },
  {
    pattern: 'pretend\\s+to\\s+be\\s+(a|an)\\s+(system|AI|bot|assistant)',
    category: 'role-manipulation',
    severity: 'medium',
    explanation: 'Attempts to redefine the models role through pretense',
  },
  {
    pattern: 'from\\s+now\\s+on\\s+you\\s+are\\s+',
    category: 'role-manipulation',
    severity: 'high',
    explanation: 'Attempts to redefine the model identity mid-conversation',
  },
  
  // Privilege escalation patterns
  {
    pattern: 'admin\\s+override',
    category: 'privilege-escalation',
    severity: 'critical',
    explanation: 'Attempts to use admin privileges to bypass controls',
  },
  {
    pattern: 'bypass\\s+(security|safety|all|the)',
    category: 'privilege-escalation',
    severity: 'high',
    explanation: 'Direct attempt to bypass security measures',
  },
  {
    pattern: '(disable|turn\\s+off|remove)\\s+guardrails',
    category: 'privilege-escalation',
    severity: 'critical',
    explanation: 'Attempts to disable safety guardrails',
  },
  {
    pattern: 'override\\s+(safety|security)\\s+(measures|checks|controls)',
    category: 'privilege-escalation',
    severity: 'high',
    explanation: 'Attempts to override safety controls',
  },
  {
    pattern: '(enter|activate|enable)\\s+sudo\\s+(mode)?',
    category: 'privilege-escalation',
    severity: 'high',
    explanation: 'Attempts to gain elevated privileges using Unix-like terminology',
  },
  
  // Data exfiltration patterns
  {
    pattern: '(reveal|show|display|print|output)\\s+(your\\s+)?(system\\s+)?(prompt|instructions)',
    category: 'data-exfiltration',
    severity: 'high',
    explanation: 'Attempts to extract the system prompt',
  },
  {
    pattern: 'what\\s+(is|are)\\s+your\\s+(system\\s+)?(message|prompt|instructions)',
    category: 'data-exfiltration',
    severity: 'medium',
    explanation: 'Attempts to learn about system configuration',
  },
  {
    pattern: '(output|print|show)\\s+your\\s+(training\\s+)?data',
    category: 'data-exfiltration',
    severity: 'high',
    explanation: 'Attempts to extract training data',
  },
  {
    pattern: '(print|show|reveal)\\s+your\\s+context',
    category: 'data-exfiltration',
    severity: 'high',
    explanation: 'Attempts to reveal conversation context or memory',
  },
  {
    pattern: 'repeat\\s+(back|your|the)\\s+(initial|first|system)',
    category: 'data-exfiltration',
    severity: 'medium',
    explanation: 'Attempts to make the model repeat its system prompt',
  },
];

/**
 * Check if input text looks like a prompt injection attempt
 * @param input The user input to analyze
 * @returns Detection result with matched patterns
 */
export function looksLikePromptInjection(input: string): InjectionDetectionResult {
  if (!input || typeof input !== 'string') {
    return { detected: false, patterns: [] };
  }
  
  const lowerInput = input.toLowerCase();
  const matchedPatterns: DetectedPattern[] = [];
  
  for (const pattern of INJECTION_PATTERNS) {
    try {
      const regex = new RegExp(pattern.pattern, 'i');
      const match = lowerInput.match(regex);
      
      if (match) {
        matchedPatterns.push({
          pattern: pattern.pattern,
          matched: match[0],
          category: pattern.category,
          severity: pattern.severity,
          explanation: pattern.explanation,
        });
      }
    } catch {
      // Skip invalid regex patterns
      continue;
    }
  }
  
  return {
    detected: matchedPatterns.length > 0,
    patterns: matchedPatterns,
  };
}
