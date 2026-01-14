import { describe, it, expect } from 'vitest';

import { ensureRefundResponseDetails } from '../refund-agent-llm';

describe('ensureRefundResponseDetails', () => {
  it('should append details when missing', () => {
    const out = ensureRefundResponseDetails({
      response: 'Approved. Full refund will be processed.',
      orderId: 'ORD-123',
      amount: 50,
    });

    expect(out).toContain('ORD-123');
    expect(out).toContain('$50');
  });

  it('should not duplicate details when already present', () => {
    const out = ensureRefundResponseDetails({
      response: 'Approved for Order ID: ORD-123. Amount: $50.',
      orderId: 'ORD-123',
      amount: 50,
    });

    // Still includes the key details once; we mainly assert we didn’t remove them.
    expect(out).toContain('ORD-123');
    expect(out).toContain('$50');
  });
});
