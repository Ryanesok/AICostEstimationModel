## ADDED Requirements

### Requirement: Trust dashboard replaces the multi-model comparison chart
The system SHALL replace the right-panel bar chart (multi-model comparison) with an estimation trust dashboard. The dashboard SHALL display the effort estimate result, a quality label, evaluation metrics, and a collapsible technical detail section. The dashboard SHALL update every time the user presses "Estimasi".

#### Scenario: User presses Estimasi with valid inputs
- **WHEN** the user enters valid feature values and presses "Estimasi"
- **THEN** the trust dashboard SHALL display: the effort estimate in person-months, a quality label, MAE and MMRE metric values, and a PRED(25) value

#### Scenario: User presses Reset Form
- **WHEN** the user presses "Reset Form"
- **THEN** the trust dashboard SHALL return to its empty/placeholder state (all metric values shown as "—")

### Requirement: Quality label is derived from MMRE thresholds
The system SHALL display a plain-language quality label next to the effort estimate. The label SHALL be determined by the auto-selected model's MMRE:
- MMRE ≤ 0.25 → "Akurasi Tinggi" (green)
- MMRE > 0.25 and ≤ 0.50 → "Cukup Baik" (yellow/amber)
- MMRE > 0.50 → "Perlu Verifikasi Manual" (red/orange)
- MMRE unavailable in metrics → "Akurasi tidak tersedia" (neutral gray)

#### Scenario: MMRE ≤ 0.25
- **WHEN** the selected model's stored MMRE is 0.20
- **THEN** the quality label SHALL display "Akurasi Tinggi" in green

#### Scenario: MMRE between 0.25 and 0.50
- **WHEN** the selected model's stored MMRE is 0.40
- **THEN** the quality label SHALL display "Cukup Baik" in amber

#### Scenario: MMRE above 0.50
- **WHEN** the selected model's stored MMRE is 0.65
- **THEN** the quality label SHALL display "Perlu Verifikasi Manual" in red/orange

#### Scenario: MMRE key absent from metrics JSON
- **WHEN** the metrics JSON does not contain an MMRE key for the selected model
- **THEN** the quality label SHALL display "Akurasi tidak tersedia" in neutral gray

### Requirement: Evaluation metrics table is displayed in the trust dashboard
The system SHALL display a compact metric table containing MAE, MMRE, and PRED(25) values for the auto-selected model. Each metric SHALL be accompanied by a brief one-line plain-language explanation.

#### Scenario: All metrics available
- **WHEN** the metrics JSON contains MAE, MMRE, and PRED(25) for the selected model
- **THEN** the table SHALL show all three values with their plain-language descriptions

#### Scenario: PRED(25) not available in metrics JSON
- **WHEN** the metrics JSON does not contain a PRED(25) key
- **THEN** the table row for PRED(25) SHALL display "N/A" without raising an error

### Requirement: Model comparison chart is removed from the default view
The system SHALL NOT display the per-model bar chart in the default result view. The chart SHALL only be available inside the collapsed "Detail Teknis" section if re-introduced, but is not required.

#### Scenario: App opens or estimate is produced
- **WHEN** the app starts or the user presses "Estimasi"
- **THEN** no multi-model comparison bar chart SHALL be visible in the main panels
