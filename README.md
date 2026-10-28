# ME/CFS cfRNA Expression Analysis

This repository contains an exploratory analysis of publicly available
circulating cell-free RNA (cfRNA) data from people with myalgic
encephalomyelitis/chronic fatigue syndrome (ME/CFS). The primary dataset is
GEO series [GSE293840](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE293840),
downloaded on September 21–22, 2026.

The project is a data-analysis project out of interest: it prepares gene-count and
sample metadata, explores cohort characteristics, performs differential
expression analysis, and examines sample structure through clustering and
visualization.

## Study

The dataset is described in:

> Gardella AE, et al. Circulating cell-free RNA signatures for the
> characterization and diagnosis of myalgic encephalomyelitis/chronic
> fatigue syndrome. *Proceedings of the National Academy of Sciences*.
> 2025;122(33):e2507345122.
> [doi:10.1073/pnas.2507345122](https://doi.org/10.1073/pnas.2507345122)

The study's analysis materials are available in the
[cfRNA_MECFS repository](https://github.com/aegardella/cfRNA_MECFS). The
`paper/` folder contains study publications and supplementary material.

## Scope and limitations

This is exploratory research code, not a clinical or diagnostic tool. The
analysis and any resulting observations have not been independently validated.
Results should not be used to guide medical decisions.

## Getting started

The analysis code and detailed instructions are in [`mecfs-explorer/`](./mecfs-explorer/).
See its [README](./mecfs-explorer/README.md) for:

- the required input files and expected data locations;
- environment setup and dependencies;
- instructions for preparing data and running analyses; and
- an overview of the analysis scripts and outputs.

## Repository layout

```text
.
├── mecfs-explorer/  # Analysis code, tests, and detailed usage instructions
└── paper/           # Related publications and supplementary material
```
