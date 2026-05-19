## ADDED Requirements

### Requirement: auto_configure warns when a feature is highly correlated with the target
`auto_configure.py` SHALL compute the Pearson correlation coefficient between each candidate feature and the selected target column; if |r| ≥ 0.95, it SHALL warn the user and require explicit confirmation before including that feature in the saved configuration.

#### Scenario: Leakage warning shown for high-correlation feature
- **WHEN** a candidate feature has |r| ≥ 0.95 with the selected target
- **THEN** the feature is flagged in the console with its correlation value
- **THEN** the user is prompted to confirm whether to keep or drop the feature
- **THEN** the feature is excluded from `features` if the user chooses to drop it

#### Scenario: No warning for safe features
- **WHEN** all candidate features have |r| < 0.95 with the selected target
- **THEN** no leakage warning is shown and the configuration flow proceeds normally

#### Scenario: Configuration not saved if all features are dropped
- **WHEN** the user drops all candidate features via the leakage check
- **THEN** `auto_configure.py` SHALL skip saving the configuration for that dataset with a clear message

### Requirement: auto_configure detects constant or near-constant columns
`auto_configure.py` SHALL detect candidate columns where the ratio of the most common value exceeds 95% of non-null rows, and SHALL exclude them automatically from the feature list with a printed notice.

#### Scenario: Constant column excluded automatically
- **WHEN** a numeric column has the same value in ≥ 95% of rows
- **THEN** it is excluded from the candidate feature list before the user sees it
- **THEN** a notice is printed identifying the excluded column and why
