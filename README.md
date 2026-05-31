# Tactical Topology: Football as a Communication Network

Analyse football team tactical structure using communication network theory
(graph resilience, max-flow, latency/tempo, pressing-as-interference), layered
with ML models and clean visualisations.

> Full README (concept explanation, quick start, figures, citation) is written
> in Phase 5. See [PLAN_1.md](PLAN_1.md) for the phased implementation plan.

## Installation

```bash
pip install -r requirements.txt
```

## Data

Built on [StatsBomb Open Data](https://github.com/statsbomb/open-data)
(free, no API key required), accessed via `statsbombpy`.
Working dataset: La Liga, Season 2015/16 (`competition_id=11`, `season_id=27`).
