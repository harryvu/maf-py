import { describe, it, expect } from 'vitest';
import { retrievePolicy, REFUND_POLICY } from '../policy';

describe('REFUND_POLICY', () => {
  it('should be a non-empty string', () => {
    expect(typeof REFUND_POLICY).toBe('string');
    expect(REFUND_POLICY.length).toBeGreaterThan(0);
  });

  it('should contain refund-related keywords', () => {
    const lowerPolicy = REFUND_POLICY.toLowerCase();
    expect(lowerPolicy).toMatch(/refund/);
  });

  it('should contain policy rules or guidelines', () => {
    const lowerPolicy = REFUND_POLICY.toLowerCase();
    // Should contain at least some policy-related content
    const hasRules = 
      lowerPolicy.includes('rule') ||
      lowerPolicy.includes('policy') ||
      lowerPolicy.includes('requirement') ||
      lowerPolicy.includes('day') ||
      lowerPolicy.includes('amount') ||
      lowerPolicy.includes('condition');
    expect(hasRules).toBe(true);
  });
});

describe('retrievePolicy', () => {
  it('should return the refund policy string', async () => {
    const policy = await retrievePolicy();
    expect(typeof policy).toBe('string');
    expect(policy.length).toBeGreaterThan(0);
  });

  it('should return the same content as REFUND_POLICY', async () => {
    const policy = await retrievePolicy();
    expect(policy).toBe(REFUND_POLICY);
  });

  it('should be an async function', () => {
    const result = retrievePolicy();
    expect(result).toBeInstanceOf(Promise);
  });

  it('should resolve successfully', async () => {
    await expect(retrievePolicy()).resolves.toBeDefined();
  });
});
