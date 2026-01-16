/**
 * Policy Module
 * Refund policy retrieval - mirrors Python refund_agent.py behavior
 */

/**
 * Refund policy constant
 * This mirrors the refund_policy.txt content from the Python agent
 */
export const REFUND_POLICY = `
# Refund Policy

## Eligibility Requirements
1. Refund requests must be made within 30 days of purchase
2. Products must be in original condition
3. Proof of purchase (order ID) is required

## Refund Amounts
- Full refund: Available for items returned within 30 days
- No refund: After 30 days

## Maximum Refund Limits
- Standard customers: Up to $500 per transaction
- Premium customers: Up to $1000 per transaction
- VIP customers: Up to $2500 per transaction

## Special Conditions
- Digital products: Non-refundable once downloaded
- Perishable items: Must be reported within 48 hours
- Custom orders: Non-refundable

## Process
1. Submit refund request with order ID
2. Provide reason for refund
3. Wait for approval (1-3 business days)
4. Refund processed to original payment method

## Contact
For questions, contact customer support.
`.trim();

/**
 * Retrieve the refund policy
 * Async to match future integration with external policy sources
 */
export async function retrievePolicy(): Promise<string> {
  // In production, this could fetch from a database or external service
  return REFUND_POLICY;
}
