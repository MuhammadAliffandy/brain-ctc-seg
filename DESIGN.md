# Brain CT Review Design Direction

## Identity

Clinical imaging review workspace for radiologists and clinicians who need to inspect a brain CT lesion, verify the proposed contour, and record a decision.

## Personality

Quiet, precise, direct, and trustworthy. The interface should resemble a focused radiology workstation rather than an analytics dashboard.

## Palette

- Canvas: near-black charcoal to preserve CT image contrast.
- Surface: one step lighter than the canvas.
- Primary text: cool white.
- Secondary text: muted blue-grey with WCAG AA contrast.
- Clinical accent: green, reserved for verified contours and the primary review action.
- Warning: amber, reserved for preview or uncertainty states.
- Error: red, reserved for failures and unsafe states.

## Typography

Use the system sans-serif stack. Headings are compact and sentence case. Measurements use tabular figures where available. Avoid decorative monospace text outside technical audit details.

## Composition

1. Case context and scan selection.
2. Large axial viewport with slice navigation.
3. Compact lesion estimate and a single primary contour-review action.
4. Additional MPR, agreement details, agent workflow, and report export on demand.

## Interaction

- Preserve the existing clinical flow and model functions.
- Keep one primary action visible at a time.
- Hide configuration that a clinician does not need during routine review.
- Always identify preview, simulated, loading, error, and clinician-edited states in text.
- Keep controls keyboard reachable with visible focus and 44px minimum touch targets.

## Liveliness Dials

- ENERGY: 2/5
- RHYTHM: 3/5
- MOTION: 1/5

The interface gets character from the imaging workspace, asymmetrical viewport-to-findings layout, and restrained clinical green. It does not use decorative gradients, glows, emoji, or ornamental cards.

