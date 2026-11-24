# Third-party sources

The official EvoTeam method repository is https://github.com/DEEP-JLU/EvoTeam. This evaluation repository stores the three dataset adapters, experiment configurations, and related data-loading, scoring, and preparation scripts. It also includes the initial agent pools and role prompts for all three datasets. Method cores, tool libraries, and frozen test pools are not distributed here.

The adapters require the `individual-benchmark-v1` runtime interface; see `methods/Evoteam/source-lock.json` for the pinned runtime commit and file checksums, and `methods/Evoteam/upstream.yaml` for interface metadata. The runtime revision is separate from the official method repository. Supply its source repository through `EVOTEAM_SOURCE_REPOSITORY` or use a verified local checkout as described in `methods/Evoteam/README.md`. Obtain GAIA, MAC-Bench, and MASBench data and environments separately under their respective licenses. MAC-Bench uses the upstream evaluator; the MASBench Game worker bridges to the upstream environment in a separate process.

The original source directory did not provide a LICENSE that could be inherited. No new license has been assigned on behalf of the authors.
