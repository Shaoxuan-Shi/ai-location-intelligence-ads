# Data

This directory separates reproducible source files from compact tables that are useful for reviewing the project.

## Layout

| Directory | Contents | Version controlled |
| --- | --- | --- |
| `raw/` | Cached Amsterdam API, OpenStreetMap, and spatial source files | No |
| `interim/` | Temporary pipeline outputs | No |
| `processed/` | Modeling-ready anchor and feature tables | Yes |

Raw files are excluded because the local cache is about 198 MB. They can be downloaded again with:

```bash
make fetch
```

The core pipeline then writes processed datasets and compact model outputs:

```bash
make rebuild
```

## Sources

- Amsterdam Crowdmonitor weekly pedestrian counts from the [Amsterdam Open Data API](https://api.data.amsterdam.nl/v1)
- Official Amsterdam shopping-area and spatial context data from the same API
- Shops, horeca, transit, tourism, lodging, and related context from [OpenStreetMap](https://www.openstreetmap.org/copyright)

The production training window contains 65 weekly dates from 2023-01-02 to 2024-03-25. Source datasets remain subject to their provider terms.

The fetch pipeline is self-contained: it uses only the documented public endpoints and files inside this repository. When a latest Crowdmonitor row lacks geometry, the pipeline reuses geometry from another row for the same API location; rows with no API geometry evidence are skipped.
