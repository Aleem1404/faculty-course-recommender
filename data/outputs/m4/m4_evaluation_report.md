# M4 Collaboration Evaluation Report

This report consolidates the technical and structural evaluation of the M4 faculty-collaboration recommendation extension. It validates internal correctness, coverage, traceability, explainability and complementarity evidence. It does not claim domain-level teaching suitability across all academic disciplines.

## Configuration

| Parameter | Value |
| --- | --- |
| Topic embedding model | all-MiniLM-L6-v2 |
| Topic normalisation similarity threshold | 0.84 |
| Minimum individual staff relevance | 0.10 |
| Maximum individual candidates per module | 20 |
| Minimum complementarity gain | 0.05 |
| Display minimum combined coverage | 0.40 |
| Display minimum complementarity gain | 0.10 |
| Semantic shared-foundation threshold | 0.80 |

## Data Preparation

| Measure | Result |
| --- | --- |
| Topic assignments processed | 18,340 |
| Staff topic profiles | 620 |
| Module topic profiles | 1,679 |
| Average topics per staff profile | 9.216 |
| Average topics per module profile | 7.520 |
| Staff metadata coverage | 100.0% |
| Module metadata coverage | 100.0% |

## Candidate Generation

| Measure | Result |
| --- | --- |
| Staff profiles loaded | 620 |
| Module profiles loaded | 1,676 |
| Modules with collaboration candidates | 120 |
| Modules without collaboration candidates | 1,556 |
| Pair candidates before output limit | 2,037 |
| Collaboration recommendations written | 1,635 |

## Structural Validation

| Validation measure | Result |
| --- | --- |
| Recommendations checked | 1,635 |
| Modules with recommendations | 120 |
| Positive complementarity rate | 100.0% |
| Topic-evidence rate | 100.0% |
| Complete-explanation rate | 100.0% |
| Record-level structural failures | 0 |
| Display-policy compliant | Yes |
| Display-policy failures | 0 |

## Ranking Results

| Metric | Minimum | Mean | Median | Maximum |
| --- | --- | --- | --- | --- |
| Collaboration score | 0.131 | 0.230 | 0.218 | 0.555 |
| Combined module coverage | 0.173 | 0.323 | 0.307 | 0.797 |
| Best individual coverage | 0.103 | 0.200 | 0.186 | 0.557 |
| Complementarity gain | 0.050 | 0.123 | 0.116 | 0.240 |
| Staff-topic overlap ratio | 0.000 | 0.022 | 0.000 | 0.333 |

The complementarity gain is calculated as:

\[	ext{ComplementarityGain}(a,b,m) = 	ext{Coverage}(a \cup b,m) - \max(	ext{Coverage}(a,m), 	ext{Coverage}(b,m))\]

A positive value means that the second staff member adds weighted module-topic coverage beyond the stronger individual candidate.

## Recommendation Outputs

| Output measure | Result |
| --- | --- |
| Recommendations written | 1,635 |
| High recommendations | 61 |
| Moderate recommendations | 1,035 |
| Exploratory recommendations | 539 |
| Strict display recommendations | 5 |
| Case studies selected | 21 |

## Semantic Foundation Analysis

| Measure | Result |
| --- | --- |
| Semantic similarity threshold | 0.80 |
| Exact shared-foundation recommendations | 460 |
| Semantic-only foundation recommendations | 22 |
| Top-ranked exact foundation rate | 40.0% |
| Top-ranked exact plus semantic foundation rate | 40.0% |
| Additional top-ranked modules supported by semantic evidence | 0 |

Semantic shared foundation was treated as supplementary explanation evidence. It did not alter the original collaboration ranking, coverage scores or complementarity gain.

## Case-Study Set

| Measure | Result |
| --- | --- |
| Case studies checked | 21 |
| Duplicate case IDs | 0 |
| Repeated module IDs across groups | 5 |

The case-study set was designed as an audit and expert-review instrument. It should not be treated as researcher-labelled ground truth for all academic disciplines.

## Key Findings

- M4 produced 1,635 collaboration recommendations across 120 modules.
- All retained recommendations had positive complementarity gain (100.0%) and traceable topic evidence (100.0%).
- All recommendations contained complete readable explanations (100.0%).
- The mean combined module coverage was 0.323, while mean best individual coverage was 0.200.
- The mean complementarity gain was 0.123.
- Structural validation recorded 0 record-level failures and the display policy was compliant.
- At semantic threshold 0.80, 22 recommendations gained semantic-only shared-foundation evidence.

## Limitations

- The recommendations are based on publicly available course and staff-profile information, which may be incomplete or outdated.
- Topic extraction and canonicalisation may retain awkward, overly broad or imperfectly ordered phrases.
- The framework does not include staff availability, workload, timetabling, contractual responsibilities, teaching preferences or institutional allocation rules.
- Structural validation confirms internal consistency and explanation traceability, not domain-level teaching suitability.
- Domain-level assessment should be performed by module leaders, subject specialists or academic managers in future work.

## Evaluation Boundary

The M4 evaluation demonstrates internal consistency, reproducible processing, positive complementarity, topic-evidence traceability and explanation completeness. It does not replace institutional decision-making or subject-specialist validation of teaching allocations.
