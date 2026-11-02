# October relevance-reject audit: concrete findings

The 920 explicit saved-text assessments cover all upstream rejects published October 1–7 UTC. Most remain novelty-unverified; targeted web checking supports only bounded claims. The original collection is unchanged. R numbers identify rows in `reject_review_ledger.csv`; D numbers identify the original direct/indirect ledger.

## Missed NVIDIA links and useful context

- **R204/R250/R272, DigitalOcean AI ecosystem:** upstream labels say not relevant, but the [October 2 company release](https://investors.digitalocean.com/news/news-details/2025/DigitalOcean-Expands-AI-Ecosystem-Launches-Partner-Program-to-Accelerate-Innovation-and-Empower-Startups-and-Builders/default.aspx) explicitly mentions NVIDIA and AMD GPU access. This supports relevance reconsideration, not an assertion of newly purchased NVIDIA capacity. First public minute remains unresolved. Repeated articles describe one development.
- **R812/R865, IREN cloud contracts:** the [October 7 company release](https://irisenergy.gcs-web.com/news-releases/news-release-details/iren-secures-new-multi-year-ai-cloud-contracts) identifies NVIDIA Blackwell deployments. The upstream rejection missed a concrete connection. A subsequently located [company wire](https://www.globenewswire.com/news-release/2025/10/7/3162322/0/en/iren-secures-new-multi-year-ai-cloud-contracts.html) displays 06:17 ET (10:17 UTC), before both saved articles. These articles are therefore follow-ups to a located earlier disclosure, not clean first-announcement records.
- **R459, NVDX:** headline names a leveraged NVIDIA-linked instrument, despite upstream rejection. The [exchange listing](https://www.cboe.com/us/equities/listings/listed_products/symbols/NVDX/) confirms the instrument linkage. The saved summary is empty. This is a relevance concern, not a verified material announcement.
- **R554, Intel September price recap:** NVIDIA investment context connects to D12/D40 and the [September 18 company announcement](https://investor.nvidia.com/news/press-release-details/2025/NVIDIA-and-Intel-to-Develop-AI-Infrastructure-and-Personal-Computing-Products/default.aspx). Relevant context can still be retrospective.
- **R10/R34/R43, Korean memory agreements:** the [October 1 OpenAI announcement](https://openai.com/index/samsung-and-sk-join-stargate/) corroborates the event also covered by D48 the next day. The first saved flash is 06:12 UTC, whereas the company page supplies only a date. Do not infer exact first availability from that page.

## Earlier reporting and genuinely different updates

- R143/R147/R149 and later October 2–3 articles follow the [Semafor original](https://www.semafor.com/article/10/01/2025/intel-amd-foundry-customer-deal), displayed October 1 at 13:58 EDT (17:58 UTC). Early foundry talks are not a signed contract. R472's later rebuttal is a potentially new response, not another instance of the same initial disclosure.
- R70 Broadcom and R91 Micas have company-wire release minutes matching saved times: [Broadcom, 13:00 UTC](https://www.globenewswire.com/news-release/2025/10/01/3159570/19933/en/Broadcom-Showcases-Industry-Leading-Quality-and-Reliability-of-Co-Packaged-Optics.html), [Micas, 14:00 UTC](https://www.globenewswire.com/news-release/2025/10/01/3159683/0/en/Micas-Networks-Achieves-4-Million-Link-Flap-Free-Hours-in-Testing-of-Industry-s-First-51-2T-Co-Packaged-Optics-Switch.html). Reliability milestones can be new while the underlying optical technology is old. These remain company claims and do not establish material NVDA impact.
- R349 Hitachi/OpenAI is corroborated by a [later October 21 release](https://www.hitachi.com/en/press/articles/2025/10/1021c/) describing an October 2 signature. That later page cannot establish the first public availability of October 3 reporting. Signature time and disclosure time must remain distinct.

## Timestamp and source-text problems

The [AMD company wire](https://www.globenewswire.com/news-release/2025/10/06/3161591/0/en/index.html) displays October 6 at 07:00 ET, or **11:00 UTC**. Two rejected records have earlier saved clocks:

| Record | Article ID | Saved UTC | Problem |
|---|---|---|---|
| R565 | 138593080 | October 6 07:28:14 | Saved market-news text contains deal details and price change before the located company release. |
| R577 | 138430710 | October 6 09:01:08 | Headline already describes AMD up over 34%, before the located company release. |

This discrepancy does **not** prove a timestamp error: prior reporting/leaks, updated versions, and timezone provenance are unresolved alternatives. Current saved text must not be treated as available at the earlier clock without resolving its history. Otherwise future information could enter a supposed predictive test.

R467 (article 136980499) has a different problem: the headline says Meta's price target is **$195**, while the summary says **$915**, and dates the reiteration September 24. Preserve that conflict; do not silently correct it or train a numeric label from it.

## What the audit does not establish

146 reject records are flagged for relevance reconsideration. Many are broad supply-chain or competition context, not demonstrated filter errors. A source URL can establish event identity, a matching release time, or earlier coverage; these are different levels of support. All records retain `first_release_proven=false`, `training_ready=false`, and `price_outcomes_used=false`. Unverified articles are not negative examples.
