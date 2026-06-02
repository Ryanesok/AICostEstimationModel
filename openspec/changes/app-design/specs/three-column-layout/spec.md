## ADDED Requirements

### Requirement: Three-column window layout
The application window SHALL use a fixed three-column layout at 1400×780 pixels with columns assigned as: left (input form), middle (model selection and results), and right (chart).

#### Scenario: App opens with three columns visible
- **WHEN** the user launches the application
- **THEN** three columns are visible simultaneously without scrolling horizontally

#### Scenario: Window is fixed size
- **WHEN** the user attempts to resize the window
- **THEN** the window SHALL NOT change dimensions

### Requirement: Left column contains dataset selector and input form
The left column SHALL contain the dataset selector dropdown and the scrollable input form; it SHALL NOT contain the model toggle, result area, or calculator.

#### Scenario: Input form is scrollable in left column
- **WHEN** the selected dataset has many input fields (e.g., COCOMO-81 with 16 fields)
- **THEN** the left column scrolls independently without affecting the middle or right columns

#### Scenario: Changing dataset updates only the left column form
- **WHEN** the user selects a different dataset from the dropdown
- **THEN** the input form in the left column is rebuilt with the new dataset's fields
- **THEN** the middle column result and calculator areas are reset to their empty state

### Requirement: Middle column contains model toggle, result, and calculator
The middle column SHALL contain: the model toggle (SVR/Linear Regression), the estimation result display, and the cost/team calculator; this content SHALL always be fully visible without scrolling.

#### Scenario: Middle column is always visible
- **WHEN** the user scrolls the left column input form
- **THEN** the middle column content SHALL remain fully visible and accessible

#### Scenario: Estimation result appears in middle column
- **WHEN** the user clicks Estimasi
- **THEN** the predicted value and unit label appear in the middle column result area
- **THEN** the calculator in the middle column automatically updates with the new prediction

### Requirement: Right column contains chart visualization
The right column SHALL contain the matplotlib bar chart comparing SVR and Linear Regression predictions; chart behavior SHALL be unchanged from the current implementation.

#### Scenario: Chart updates after estimation
- **WHEN** the user clicks Estimasi and prediction succeeds
- **THEN** the right column chart updates to show both model predictions as horizontal bars

#### Scenario: Empty chart shown before estimation
- **WHEN** the application first loads or after Reset Form
- **THEN** the right column shows the empty chart placeholder text
