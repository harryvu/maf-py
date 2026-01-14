/**
 * Pattern Detector
 * Comprehensive pattern detection for prompt injection attacks
 */

import type { InjectionCategory, RiskLevel, DetectedPattern } from '../types/security';

export interface PatternDefinition {
  /** Regex pattern string */
  pattern: string;
  /** Category of injection */
  category: InjectionCategory;
  /** Severity level */
  severity: RiskLevel;
  /** Educational explanation */
  explanation: string;
}

export interface DetectionResult {
  /** Whether any pattern was detected */
  detected: boolean;
  /** List of matched patterns with details */
  patterns: DetectedPattern[];
}

/**
 * Default patterns covering 25+ injection techniques
 */
const DEFAULT_PATTERNS: PatternDefinition[] = [
  // System prompt override (5 patterns)
  {
    pattern: 'ignore\\s+(all\\s+)?(previous|prior|above)\\s+instructions',
    category: 'system-prompt-override',
    severity: 'high',
    explanation: 'Attempts to override the system prompt by instructing the model to ignore its original instructions.',
  },
  {
    pattern: 'disregard\\s+(your|all|any)\\s+(instructions|rules)',
    category: 'system-prompt-override',
    severity: 'high',
    explanation: 'Attempts to make the model disregard its guidelines.',
  },
  {
    pattern: 'forget\\s+(what|everything|all)\\s+(you\\s+were|you\\s+have)',
    category: 'system-prompt-override',
    severity: 'high',
    explanation: 'Attempts to reset model context and memory.',
  },
  {
    pattern: '</system>',
    category: 'system-prompt-override',
    severity: 'critical',
    explanation: 'Uses closing system tag to break out of prompt structure.',
  },
  {
    pattern: '\\[INST\\]|\\[/INST\\]',
    category: 'system-prompt-override',
    severity: 'critical',
    explanation: 'Uses Llama-style instruction tags to inject new instructions.',
  },
  
  // Jailbreak patterns (5 patterns)
  {
    pattern: '\\bDAN\\b',
    category: 'jailbreak',
    severity: 'critical',
    explanation: 'DAN (Do Anything Now) is a well-known jailbreak technique.',
  },
  {
    pattern: 'Do\\s+Anything\\s+Now',
    category: 'jailbreak',
    severity: 'critical',
    explanation: 'Explicit DAN jailbreak attempt.',
  },
  {
    pattern: '(have\\s+)?no\\s+restrictions',
    category: 'jailbreak',
    severity: 'high',
    explanation: 'Attempts to remove model restrictions.',
  },
  {
    pattern: 'hypothetically\\s+(if|what|speaking)',
    category: 'jailbreak',
    severity: 'medium',
    explanation: 'Uses hypothetical framing to bypass safety measures.',
  },
  {
    pattern: '(enable|activate|enter)\\s+developer\\s+mode',
    category: 'jailbreak',
    severity: 'high',
    explanation: 'Attempts to activate a fake developer mode.',
  },
  
  // Role manipulation (5 patterns)
  {
    pattern: 'act\\s+as\\s+(an?\\s+)?admin',
    category: 'role-manipulation',
    severity: 'high',
    explanation: 'Attempts to assume admin privileges through roleplay.',
  },
  {
    pattern: 'you\\s+are\\s+(now\\s+)?a\\s+different\\s+(AI|assistant)',
    category: 'role-manipulation',
    severity: 'high',
    explanation: 'Attempts to change the model identity.',
  },
  {
    pattern: 'roleplay\\s+(as|like)',
    category: 'role-manipulation',
    severity: 'medium',
    explanation: 'Uses roleplay to potentially bypass restrictions.',
  },
  {
    pattern: 'pretend\\s+to\\s+be',
    category: 'role-manipulation',
    severity: 'medium',
    explanation: 'Attempts to change model behavior through pretense.',
  },
  {
    pattern: 'from\\s+now\\s+on\\s+you\\s+are',
    category: 'role-manipulation',
    severity: 'high',
    explanation: 'Attempts to redefine model identity mid-conversation.',
  },
  
  // Privilege escalation (5 patterns)
  {
    pattern: 'admin\\s+override',
    category: 'privilege-escalation',
    severity: 'critical',
    explanation: 'Attempts to use admin privileges to bypass controls.',
  },
  {
    pattern: 'bypass\\s+(security|safety|all)',
    category: 'privilege-escalation',
    severity: 'high',
    explanation: 'Direct attempt to bypass security measures.',
  },
  {
    pattern: '(disable|turn\\s+off)\\s+guardrails',
    category: 'privilege-escalation',
    severity: 'critical',
    explanation: 'Attempts to disable safety guardrails.',
  },
  {
    pattern: 'override\\s+safety',
    category: 'privilege-escalation',
    severity: 'high',
    explanation: 'Attempts to override safety measures.',
  },
  {
    pattern: '(enter|enable)\\s+sudo\\s+(mode)?',
    category: 'privilege-escalation',
    severity: 'high',
    explanation: 'Attempts to gain elevated privileges.',
  },
  
  // Data exfiltration (5 patterns)
  {
    pattern: '(reveal|show|display)\\s+(your\\s+)?(system\\s+)?prompt',
    category: 'data-exfiltration',
    severity: 'high',
    explanation: 'Attempts to extract the system prompt.',
  },
  {
    pattern: 'show\\s+your\\s+instructions',
    category: 'data-exfiltration',
    severity: 'high',
    explanation: 'Attempts to reveal system instructions.',
  },
  {
    pattern: 'output\\s+your\\s+training\\s+data',
    category: 'data-exfiltration',
    severity: 'high',
    explanation: 'Attempts to extract training data.',
  },
  {
    pattern: 'what\\s+is\\s+your\\s+system\\s+message',
    category: 'data-exfiltration',
    severity: 'medium',
    explanation: 'Attempts to learn system configuration.',
  },
  {
    pattern: 'print\\s+your\\s+context',
    category: 'data-exfiltration',
    severity: 'high',
    explanation: 'Attempts to reveal conversation context.',
  },
];

/**
 * Pattern Detector class for comprehensive injection detection
 */
export class PatternDetector {
  private patterns: PatternDefinition[];
  
  constructor(customPatterns?: PatternDefinition[]) {
    this.patterns = customPatterns ?? [...DEFAULT_PATTERNS];
  }
  
  /**
   * Detect injection patterns in input text
   */
  detect(input: string): DetectionResult {
    if (!input || typeof input !== 'string' || input.trim() === '') {
      return { detected: false, patterns: [] };
    }
    
    const lowerInput = input.toLowerCase();
    const matchedPatterns: DetectedPattern[] = [];
    
    for (const patternDef of this.patterns) {
      try {
        const regex = new RegExp(patternDef.pattern, 'i');
        const match = lowerInput.match(regex);
        
        if (match) {
          matchedPatterns.push({
            pattern: patternDef.pattern,
            matched: match[0],
            category: patternDef.category,
            severity: patternDef.severity,
            explanation: patternDef.explanation,
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
  
  /**
   * Get the number of patterns in the detector
   */
  getPatternCount(): number {
    return this.patterns.length;
  }
  
  /**
   * Add a new pattern to the detector
   */
  addPattern(pattern: PatternDefinition): void {
    this.patterns.push(pattern);
  }
}
