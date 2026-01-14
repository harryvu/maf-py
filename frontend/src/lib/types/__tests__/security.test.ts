import { describe, it, expect } from 'vitest';
import {
  SecurityAnalysis,
  DetectedPattern,
  InjectionCategory,
  RiskLevel,
  isValidSecurityAnalysis,
  createSecurityAnalysis,
} from '../security';

describe('SecurityAnalysis', () => {
  describe('InjectionCategory', () => {
    it('should include system-prompt-override category', () => {
      const category: InjectionCategory = 'system-prompt-override';
      expect(category).toBe('system-prompt-override');
    });

    it('should include jailbreak category', () => {
      const category: InjectionCategory = 'jailbreak';
      expect(category).toBe('jailbreak');
    });

    it('should include data-exfiltration category', () => {
      const category: InjectionCategory = 'data-exfiltration';
      expect(category).toBe('data-exfiltration');
    });

    it('should include privilege-escalation category', () => {
      const category: InjectionCategory = 'privilege-escalation';
      expect(category).toBe('privilege-escalation');
    });

    it('should include role-manipulation category', () => {
      const category: InjectionCategory = 'role-manipulation';
      expect(category).toBe('role-manipulation');
    });

    it('should include indirect-injection category', () => {
      const category: InjectionCategory = 'indirect-injection';
      expect(category).toBe('indirect-injection');
    });
  });

  describe('RiskLevel', () => {
    it('should include low risk level', () => {
      const level: RiskLevel = 'low';
      expect(level).toBe('low');
    });

    it('should include medium risk level', () => {
      const level: RiskLevel = 'medium';
      expect(level).toBe('medium');
    });

    it('should include high risk level', () => {
      const level: RiskLevel = 'high';
      expect(level).toBe('high');
    });

    it('should include critical risk level', () => {
      const level: RiskLevel = 'critical';
      expect(level).toBe('critical');
    });
  });

  describe('DetectedPattern', () => {
    it('should have required properties', () => {
      const pattern: DetectedPattern = {
        pattern: 'ignore.*instructions',
        matched: 'ignore all previous instructions',
        category: 'system-prompt-override',
        severity: 'high',
        explanation: 'Attempts to override system prompt',
      };
      expect(pattern.pattern).toBe('ignore.*instructions');
      expect(pattern.matched).toBe('ignore all previous instructions');
      expect(pattern.category).toBe('system-prompt-override');
      expect(pattern.severity).toBe('high');
      expect(pattern.explanation).toBe('Attempts to override system prompt');
    });
  });

  describe('isValidSecurityAnalysis', () => {
    it('should return true for valid security analysis', () => {
      const validAnalysis: SecurityAnalysis = {
        isInjectionDetected: true,
        riskLevel: 'high',
        detectedPatterns: [
          {
            pattern: 'ignore.*instructions',
            matched: 'ignore all previous instructions',
            category: 'system-prompt-override',
            severity: 'high',
            explanation: 'Attempts to override system prompt',
          },
        ],
        owaspCategory: 'LLM01',
        educationalNotes: ['This is a prompt injection attempt'],
      };
      expect(isValidSecurityAnalysis(validAnalysis)).toBe(true);
    });

    it('should return true for analysis with no injection detected', () => {
      const safeAnalysis: SecurityAnalysis = {
        isInjectionDetected: false,
        riskLevel: 'low',
        detectedPatterns: [],
        educationalNotes: [],
      };
      expect(isValidSecurityAnalysis(safeAnalysis)).toBe(true);
    });

    it('should return false for null', () => {
      expect(isValidSecurityAnalysis(null)).toBe(false);
    });

    it('should return false for undefined', () => {
      expect(isValidSecurityAnalysis(undefined)).toBe(false);
    });

    it('should return false when isInjectionDetected is missing', () => {
      const invalidAnalysis = {
        riskLevel: 'high',
        detectedPatterns: [],
        educationalNotes: [],
      };
      expect(isValidSecurityAnalysis(invalidAnalysis)).toBe(false);
    });

    it('should return false when riskLevel is missing', () => {
      const invalidAnalysis = {
        isInjectionDetected: false,
        detectedPatterns: [],
        educationalNotes: [],
      };
      expect(isValidSecurityAnalysis(invalidAnalysis)).toBe(false);
    });

    it('should return false when detectedPatterns is missing', () => {
      const invalidAnalysis = {
        isInjectionDetected: false,
        riskLevel: 'low',
        educationalNotes: [],
      };
      expect(isValidSecurityAnalysis(invalidAnalysis)).toBe(false);
    });

    it('should return false when riskLevel is invalid', () => {
      const invalidAnalysis = {
        isInjectionDetected: false,
        riskLevel: 'unknown',
        detectedPatterns: [],
        educationalNotes: [],
      };
      expect(isValidSecurityAnalysis(invalidAnalysis)).toBe(false);
    });
  });

  describe('createSecurityAnalysis', () => {
    it('should create a safe analysis with empty patterns', () => {
      const analysis = createSecurityAnalysis([]);
      expect(analysis.isInjectionDetected).toBe(false);
      expect(analysis.riskLevel).toBe('low');
      expect(analysis.detectedPatterns).toEqual([]);
      expect(analysis.educationalNotes).toEqual([]);
    });

    it('should create analysis with detected patterns', () => {
      const patterns: DetectedPattern[] = [
        {
          pattern: 'ignore.*instructions',
          matched: 'ignore all previous instructions',
          category: 'system-prompt-override',
          severity: 'high',
          explanation: 'Attempts to override system prompt',
        },
      ];
      const analysis = createSecurityAnalysis(patterns);
      expect(analysis.isInjectionDetected).toBe(true);
      expect(analysis.riskLevel).toBe('high');
      expect(analysis.detectedPatterns).toEqual(patterns);
    });

    it('should set riskLevel to highest severity pattern', () => {
      const patterns: DetectedPattern[] = [
        {
          pattern: 'pattern1',
          matched: 'match1',
          category: 'jailbreak',
          severity: 'medium',
          explanation: 'Medium risk',
        },
        {
          pattern: 'pattern2',
          matched: 'match2',
          category: 'data-exfiltration',
          severity: 'critical',
          explanation: 'Critical risk',
        },
      ];
      const analysis = createSecurityAnalysis(patterns);
      expect(analysis.riskLevel).toBe('critical');
    });

    it('should set owaspCategory to LLM01 when injection detected', () => {
      const patterns: DetectedPattern[] = [
        {
          pattern: 'ignore.*instructions',
          matched: 'ignore all previous instructions',
          category: 'system-prompt-override',
          severity: 'high',
          explanation: 'Attempts to override system prompt',
        },
      ];
      const analysis = createSecurityAnalysis(patterns);
      expect(analysis.owaspCategory).toBe('LLM01');
    });
  });
});
