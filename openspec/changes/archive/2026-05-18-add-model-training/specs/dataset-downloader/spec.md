## ADDED Requirements

### Requirement: URL registry file
The system SHALL maintain a plain-text file `datasets.txt` at the project root where each non-empty, non-comment line is a raw CSV URL to download.

#### Scenario: Valid URL line
- **WHEN** a line in `datasets.txt` is a valid URL string not starting with `#`
- **THEN** `downloader.py` attempts to download that URL

#### Scenario: Comment and blank lines ignored
- **WHEN** a line starts with `#` or is blank
- **THEN** `downloader.py` skips it silently

---

### Requirement: Download CSV files into data/
The downloader SHALL fetch each URL and save the file to `data/<filename>` where `<filename>` is extracted from the URL path (the last path segment, e.g. `cocomo81.csv`).

#### Scenario: Successful download
- **WHEN** the URL is reachable and returns a 200 response
- **THEN** the file is saved to `data/<filename>` and a success message is printed

#### Scenario: File already exists — skip
- **WHEN** `data/<filename>` already exists on disk
- **THEN** the downloader skips the download and prints a skip notice; the existing file is NOT overwritten

---

### Requirement: Fault-tolerant on network errors
The downloader SHALL wrap each download in a try/except block. Any network-level exception (timeout, connection error, HTTP error) SHALL be caught, logged as a WARNING, and the script SHALL continue to the next URL without crashing.

#### Scenario: Network error on one URL
- **WHEN** a download fails with any exception
- **THEN** the downloader prints a WARNING with the URL and error, then proceeds to the next URL

#### Scenario: All URLs fail
- **WHEN** every URL in `datasets.txt` raises an exception
- **THEN** the script exits normally (exit code 0) after logging each failure
