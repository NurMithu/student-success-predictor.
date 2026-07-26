# Dataset

**File:** `students_dropout_academic_success.csv`

## Source

**Predict Students' Dropout and Academic Success**
UCI Machine Learning Repository — Dataset ID 697

> Realinho, V., Vieira Martins, M., Machado, J., & Baptista, L. (2021).
> *Predict Students' Dropout and Academic Success* [Dataset]. UCI Machine
> Learning Repository. https://doi.org/10.24432/C5MC89

Also described in:

> Martins, M. V., Tolledo, D., Machado, J., Baptista, L. M. T., & Realinho, V. (2021).
> Early prediction of student's performance in higher education: A case study.
> *Trends and Applications in Information Systems and Technologies.*

License: **CC BY 4.0**

## Description

Real, de-identified data from a Portuguese higher-education institution, covering
students enrolled in undergraduate degrees such as agronomy, design, education,
nursing, journalism, management, social service, and technologies.

- **Rows:** 4,424 students
- **Columns:** 36 raw features + 1 target
- **Target classes:** `Dropout` (1,421) · `Enrolled` (794) · `Graduate` (2,209)
- **Missing values:** none (pre-cleaned by the dataset authors)

Features fall into four groups:
1. **Demographics** — marital status, nationality, gender, age at enrollment, displaced/special-needs status
2. **Socio-economic** — parents' qualification & occupation, scholarship, debtor status, tuition status
3. **Academic path** — application mode/order, course, previous qualification, attendance
4. **Academic performance** — units credited/enrolled/evaluated/approved and average grade for semesters 1 and 2
5. **Macroeconomic context** — unemployment rate, inflation rate, GDP at time of enrollment

## Why this dataset

It is one of the most widely used, peer-reviewed, real-world benchmarks for
student dropout/success prediction — donated to UCI in 2021 and featured in
numerous academic papers — making it a credible foundation for a reproducible
prediction system.

## Reproducing the download

The file in this folder is a straight copy of the original UCI CSV (also
mirrored on Kaggle). To fetch it yourself:

```python
# Option A: via the UCI helper package
pip install ucimlrepo
from ucimlrepo import fetch_ucirepo
data = fetch_ucirepo(id=697)

# Option B: direct download
# https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success
```
