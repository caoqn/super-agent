# Third-party sources

EvoTeam method source is provided by the external repository https://github.com/caoqn/EvoTeam. This evaluation repository stores the three dataset adapters, experiment configurations, and related data-loading, scoring, and preparation scripts. It also includes the initial agent pools and role prompts for all three datasets. Method cores, tool libraries, and frozen test pools are not distributed here.

The adapters originate from individual-dataset experiment implementations. Their source repository is the `individual-benchmark-v1` branch; see `methods/Evoteam/source-lock.json` for the pinned commit and file checksums, and `methods/Evoteam/upstream.yaml` for interface metadata. Obtain GAIA, MAC-Bench, and MASBench data and environments separately under their respective licenses. MAC-Bench uses the upstream evaluator; the MASBench Game worker bridges to the upstream environment in a separate process.

The original source directory did not provide a LICENSE that could be inherited. No new license has been assigned on behalf of the authors.
