# Changelog

## 0.2.1

The first public release.

- Data: `tf.pull` loads a series over a range of UTC days from Polymarket, Kalshi or Hyperliquid into pandas. It splits the range into the requests each venue accepts, reuses what you already have (the local cache, then your account's ready jobs), downloads with resume and returns `l1`, `l2` and `trades` with a `date` column. Also `tf.search`, `tf.me`, `tf.plan_pull` (what a pull would cost, requesting nothing), `tf.known_gaps` and `tf.sample` (the public Polymarket sample day, no key).
- Execution cost: `tf.LocalReplay` replays a downloaded day with the TickFoundry Lab's own code: `simulate` (an immediate order walked against the recorded book), `cost_to_move`, a labelled fee estimate, and the wrappers `tf.round_trip` and `tf.sliced_order` (a lower bound).
- The book over time: `LocalReplay.series`, `candles` and `snapshots`.
- Venues: Polymarket and Hyperliquid throughout; Kalshi for data, replay and simulation. The Kalshi fee estimate is marked unverified and warns when used.

Kalshi replay and simulation run in this package on downloaded Kalshi files. The hosted Lab adds Kalshi by the end of November 2026.
