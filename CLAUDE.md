# ATI Course Project 2026 — DataSEC track

This workspace is track 3: DataSEC, environmental sound classification under noise (audio).

- Project brief: `docs/03-datasec-environmental-audio.pdf`
- Scoring & submission guide: `guide/student-guide.pdf` (summarized below)
- Raw dataset: `DATASEC/` (22 class folders of .wav files)
- Report drafts: `research/`

## Scoring & Submission Guide (from guide/student-guide.pdf)

The guide applies to all four course tracks; each group works on one track all semester.

### Grading weights

| Component | Weight | What it measures |
|---|---|---|
| Attendance | 10% | Lecture attendance (8 points) and tutorial participation (2 points: active work in sessions, question engagement, checkpoint completion) |
| Midterm | 30% | Presentation: dataset understanding, preprocessing, baseline modeling, error analysis |
| Final | 60% | Paper-style report, viva, research-question experiments, reproducibility |

### Midterm: expected evidence (presentation checkpoint)

- dataset description and access protocol
- exploratory data analysis with task-appropriate statistics
- train/validation/test split and its justification
- preprocessing pipeline
- baseline or pretrained model results
- correct metrics for the task
- result tables and at least one error-analysis view
- proposed Final research question

A clean, correct baseline study is better than many poorly controlled experiments.

### Final: expected evidence (paper-style report + viva)

Must go beyond the Midterm and answer research questions with controlled experiments.

- paper-style report with abstract, introduction, data, method, experiments, results, discussion, limitations, conclusion
- viva defense of design decisions and results
- controlled experiments designed to answer the required research question
- prediction-level outputs or saved probabilities
- tables and figures generated from code
- reproducible notebook or code with a clear README
- explicit limitations and failure-case analysis

The Final grade depends on whether the experiments actually answer the research question, not only on whether a model scores high.

### Submission package (repository or archive)

- report as PDF
- presentation slides (if requested)
- executable notebook or code
- README with setup and run instructions
- fixed split IDs, or scripts that reproduce the same split
- configuration files or clearly documented hyperparameters
- generated result tables and figures
- prediction-level CSV or saved probabilities (required by each brief)
- note listing hardware, package versions, and random seeds

Do not submit raw restricted datasets unless the dataset license explicitly permits redistribution.

### Academic integrity

- Follow the license and terms of each dataset.
- Do not upload restricted raw data to public repositories.
- Do not tune models on the test set.
- Do not report results that cannot be reproduced from submitted code.
- Do not fabricate experiments, metrics, figures, or failure examples.
- Cite all pretrained models, papers, datasets, and code used.

### What strong projects show

- a simple but valid experimental design
- clean split discipline (no test-set leakage)
- task-appropriate metrics reported correctly
- comparison across models or protocols
- thoughtful failure analysis
- reproducible result generation
- a clear, evidence-backed answer to at least one research question
- honest limitations instead of overclaiming
