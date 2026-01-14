import { describe, it, expect } from 'vitest';
import { 
  EducationalSettings, 
  DEFAULT_SETTINGS,
  isValidEducationalSettings 
} from '../settings';

describe('EducationalSettings', () => {
  describe('DEFAULT_SETTINGS', () => {
    it('should have guardrails enabled by default', () => {
      expect(DEFAULT_SETTINGS.guardrailsEnabled).toBe(true);
    });

    it('should have simulation mode enabled by default', () => {
      expect(DEFAULT_SETTINGS.simulationMode).toBe(true);
    });

    it('should have admin bypass disabled by default', () => {
      expect(DEFAULT_SETTINGS.adminBypass).toBe(false);
    });

    it('should have all required properties defined', () => {
      expect(DEFAULT_SETTINGS).toHaveProperty('guardrailsEnabled');
      expect(DEFAULT_SETTINGS).toHaveProperty('simulationMode');
      expect(DEFAULT_SETTINGS).toHaveProperty('adminBypass');
    });
  });

  describe('isValidEducationalSettings', () => {
    it('should return true for valid settings object', () => {
      const validSettings: EducationalSettings = {
        guardrailsEnabled: true,
        simulationMode: true,
        adminBypass: false,
      };
      expect(isValidEducationalSettings(validSettings)).toBe(true);
    });

    it('should return false for null', () => {
      expect(isValidEducationalSettings(null)).toBe(false);
    });

    it('should return false for undefined', () => {
      expect(isValidEducationalSettings(undefined)).toBe(false);
    });

    it('should return false for non-object types', () => {
      expect(isValidEducationalSettings('string')).toBe(false);
      expect(isValidEducationalSettings(123)).toBe(false);
      expect(isValidEducationalSettings([])).toBe(false);
    });

    it('should return false when guardrailsEnabled is missing', () => {
      const invalidSettings = {
        simulationMode: true,
        adminBypass: false,
      };
      expect(isValidEducationalSettings(invalidSettings)).toBe(false);
    });

    it('should return false when simulationMode is missing', () => {
      const invalidSettings = {
        guardrailsEnabled: true,
        adminBypass: false,
      };
      expect(isValidEducationalSettings(invalidSettings)).toBe(false);
    });

    it('should return false when adminBypass is missing', () => {
      const invalidSettings = {
        guardrailsEnabled: true,
        simulationMode: true,
      };
      expect(isValidEducationalSettings(invalidSettings)).toBe(false);
    });

    it('should return false when properties have wrong types', () => {
      const invalidSettings = {
        guardrailsEnabled: 'true',
        simulationMode: 1,
        adminBypass: 'no',
      };
      expect(isValidEducationalSettings(invalidSettings)).toBe(false);
    });
  });
});
