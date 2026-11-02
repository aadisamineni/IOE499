# Follow-up on the six announcement candidates

Checked 2026-10-26. This is publication-timing research, not a price-reaction analysis. No tick files or minute caches were available for these measurements. Public reporting sometimes includes price commentary; it was not used as a measured event-study outcome.

## Findings

All clock times below are UTC. A located release/report timestamp is not proof that nobody disclosed the information earlier. Current web pages may have been revised, and minute-level display precision is not tick-level availability evidence.

| Review ID | Development | Dataset article time | Located evidence time | Interpretation |
|---|---|---|---|---|
| 3 | PEGATRON OCP product showcase | Oct 13, 2025 15:00 | Oct 13, 2025 15:00 | Company-issued PR Newswire release matches timestamp; good publication anchor, not proof every showcased product was newly disclosed. |
| 5 | Qualcomm AI200/AI250 launch | Oct 27, 2025 14:34:58 | Oct 27, 2025 14:00 | Reuters coverage existed about 35 minutes earlier. Sample is follow-up coverage of an announcement, not first-release evidence. |
| 14 | AWS Trainium3 availability | Dec 2, 2025 19:17 | Dec 2, 2025 16:00 | A time-stamped report establishes earlier public coverage, 3h17m before sample. Company release confirms development, but its exact release minute was not established. |
| 23 | Microsoft Maia 200 launch | Jan 26, 2026 16:03:12 | Jan 26, 2026 16:02 | Matching Reuters syndication displays 11:02 EST, 72 seconds before saved item. This establishes nearby coverage, not the original company release minute. |
| 34 | OpenAI funding involving NVIDIA | Feb 27, 2026 15:48:40 | Feb 27, 2026 14:13 | Time-stamped coverage predates sample by 1h35m40s. Company announcement confirms development; exact first public release still unverified. |
| 43 | HP workstation product launch | Mar 24, 2026 13:30 | Mar 24, 2026 13:30 | HP-issued GlobeNewswire release matches timestamp. Opening-bell coincidence makes a simple NVDA reaction especially hard to interpret. |

## Evidence

### 3 — PEGATRON

[Company-issued PR Newswire release](https://www.prnewswire.com/news-releases/pegatron-powers-the-future-of-data-centers-with-advanced-ocp-solutions-at-ocp-global-summit-2025-302581970.html) displays October 13, 2025, 11:00 ET (EDT, UTC−4). Title and described platform lineup match the sampled record. [Company newsroom](https://svr.pegatroncorp.com/news/301) corroborates the release. Suitable for a pilot around a known release timestamp; further product-history review is needed to distinguish genuinely new specifications from a previously known showcase.

### 5 — Qualcomm

[Reuters story syndicated by StreetInsider](https://www.streetinsider.com/Reuters/Qualcomm%2Bannounces%2Bnew%2BAI%2Bchips%2Bin%2Bdata%2Bcenter%2Bpush%2C%2Bshares%2Bsurge/25504129.html) displays October 27, 2025, 10:00 AM EDT. Its described chips and Humain commitment match the underlying development. [Industry reproduction of the company release](https://www.edge-ai-vision.com/2025/10/qualcomm-unveils-ai200-and-ai250-redefining-rack-scale-data-center-inference-performance-for-the-ai-era/) dates it October 27. The current [Qualcomm page](https://www.qualcomm.com/news/releases/2025/10/qualcomm-unveils-ai200-and-ai250-redefining-rack-scale-data-cent) displays October 28. Preserve that discrepancy; October 27 public coverage shows that the later displayed company date must not be treated as first availability. For an exploratory chart, 14:00 UTC is a located public-report anchor, not a certified first-release timestamp.

### 14 — Amazon

[Company announcement](https://press.aboutamazon.com/2025/12/trainium3-ultraservers-now-available-enabling-customers-to-train-and-deploy-ai-models-faster-at-lower-cost) confirms availability on December 2. [Time-stamped coverage](https://techcrunch.com/2025/12/02/amazon-releases-an-impressive-new-ai-chip-and-teases-a-nvidia-friendly-roadmap/) is displayed at 08:00 PST, establishing public coverage by 16:00 UTC. The sampled article is later commentary. The broader conference also included NVIDIA-related collaboration news: competing and complementary announcements may overlap, preventing attribution of an NVDA move to Trainium3 alone.

### 23 — Microsoft

[Company announcement](https://news.microsoft.com/source/emea/2026/01/microsoft-introduces-maia-200-new-inference-accelerator-enhances-ai-performance-in-azure/) confirms the January 26 launch. [Matching Reuters syndication](https://www.streetinsider.com/Reuters/Microsoft%2Brolls%2Bout%2Bnext%2Bgeneration%2Bof%2Bits%2BAI%2Bchips%2C%2Btakes%2Baim%2Bat%2BNvidia%27s%2Bsoftware/25891688.html) displays 11:02 AM EST in the searchable page record; a subsequent direct fetch was unavailable. Other sources suggest 16:00 coverage, but their timezone/provenance was not sufficiently established here to call it a verified original-release minute. Prioritize locating the original Microsoft page metadata or historical Reuters wire record. The 16:02 UTC anchor is provisional and has weaker retrieval evidence than the two company-wire anchors.

### 34 — OpenAI

[Company announcement](https://openai.com/index/scaling-ai-for-everyone/) confirms the funding. [Earlier coverage](https://techcrunch.com/2026/02/27/openai-raises-110b-in-one-of-the-largest-private-funding-rounds-in-history/) displays 06:13 PST. That establishes coverage before the sampled Bloomberg item, not the first public release. Earlier leaks, commitments and funding expectations need separate review before treating the full reported amount as a surprise.

### 43 — HP

[HP-issued GlobeNewswire release](https://www.globenewswire.com/news-release/2026/03/24/3261401/0/en/hp-imagine-2026-hp-announces-high-performance-pcs-for-the-most-demanding-local-compute-and-ai-workloads.html) displays March 24, 2026, 09:30 ET (EDT, UTC−4), matching the saved record. [HP newsroom](https://www.hp.com/us-en/newsroom/press-releases/2026/hp-pc-local-compute-ai-workloads.html) corroborates it. This is mainly an HP product announcement mentioning NVIDIA components, not necessarily material new information about NVIDIA's overall business. At the opening bell there is no preceding regular-session price interval: include premarket quotes and overnight context or analyze it separately.

## What can be done without the 80 GB archive?

The timing investigation above is possible without price data. Measuring returns is not. For a first descriptive check, export the existing AVS minute-quote caches for these six dates rather than transfer all ticks. The builder stores them under:

```text
tick-news-labeling_arc/output/.minute_cache/NVDA/NVDA_minute_quotes_YYYY-MM-DD.parquet
```

Export the six whole days (or a small window covering 15 minutes before the evidence anchor through 30 minutes after the later dataset publication). Full days are simpler and let us inspect other nearby news without repeatedly moving files. Six regular sessions contain roughly 2,340 minute observations before missingness, far smaller than the raw archive. Keep premarket for HP's opening-bell event. Useful columns are `minute_end_utc`, `midpoint`, `bid_price`, `ask_price`, `spread_bps`, `last_quote_utc`, and `valid_quote_updates`. These daily caches use unprefixed columns; the builder's combined in-memory frame prefixes them with the symbol.

QQQ/SOXX minute data over the same windows would allow a market comparison; NVDA alone permits only raw price paths. Preserve timestamps/timezones and quote-staleness fields. Existing endpoint caches can show minute-scale movement but cannot establish sub-minute ordering or fully reconstruct intraminute volatility. Do not call squared minute-close returns tick-based realized volatility.

Use `six_candidate_timing.csv` as the manifest. Its evidence anchors are for exploratory charts. Keep original dataset publication times as separate anchors: the original-release reaction and the delayed article prediction are different experiments. Do not silently relabel all training articles with the earlier time, because their later text may not have existed then.

Once the small export is available, plot both anchors, calculate descriptive 5/15/30-minute midpoint returns with freshness checks, inspect competing news and market movement, and report uncertainty. Six hand-reviewed cases are a feasibility pilot, not a statistical demonstration of predictive power or causality.
