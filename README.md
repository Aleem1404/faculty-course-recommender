# Faculty-Course Recommender

An explainable recommendation system for matching academic staff expertise
to university modules and identifying cross-departmental teaching
collaboration opportunities.

## Main functions

- Rank academic staff for primary module delivery.
- Recommend staff from other departments for collaborative lectures.
- Use staff profiles, research interests, publications and teaching evidence.
- Enrich publication evidence using available metadata links.
- Construct an explainable knowledge graph.
- Display shared topics, graph paths and written explanations.

## Initial methods

- TF-IDF and cosine similarity baseline.
- Sentence-transformer semantic ranking.
- Knowledge-graph-enhanced ranking.
- Cross-department collaboration ranking.
- Evidence-based explanation generation.

## Interface

The first interface allows users to select an existing course and module.
Custom teaching-content entry is outside the first implementation.