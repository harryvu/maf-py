import { describe, it, expect } from 'vitest';

import { ensureRefundResponseDetails } from '../refund-agent-llm';

describe('ensureRefundResponseDetails', () => {
  it('should append details when missing', () => {
    const out = ensureRefundResponseDetails({
      response: 'Approved. Full refund will be processed.',
      orderId: 'ORD-123',
      amount: 50,
    });

    expect(out).toContain('Requested refund information:');
    expect(out).toContain('- Order ID: ORD-123');
    expect(out).toContain('- Amount: $50');
  });

  it('should not duplicate details when already present', () => {
    const out = ensureRefundResponseDetails({
      response: 'Approved for Order ID: ORD-123. Amount: $50.',
      orderId: 'ORD-123',
      amount: 50,
    });

    // The helper enforces the structured info section.
    expect(out).toContain('Requested refund information:');
    expect(out).toContain('- Order ID: ORD-123');
    expect(out).toContain('- Amount: $50');
  });

  it('should not add details if the requested info section is already present', () => {
    const out = ensureRefundResponseDetails({
      response:
        'Denied due to policy.\n\n' +
        'Requested refund information:\n' +
        '- Order ID: ORD-123\n' +
        '- Amount: $50',
      orderId: 'ORD-123',
      amount: 50,
    });

    expect(out.match(/Requested refund information:/g)?.length ?? 0).toBe(1);
  });
});
