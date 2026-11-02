# Earlier coverage in the full saved dataset

Follow-up 2026-10-26. The 50 sampled articles were not the whole collection. Finding an earlier public source did not establish that it was missing from the collection. Previous wording that suggested missing earliest coverage was stronger than the evidence supported.

Searched the full 53,073-row classified news CSV, including excluded records, for four developments in bounded date windows. Matches and filtering fields are in `earlier_coverage_matches.csv`. Keyword matches are candidate coverage, not a complete semantic event grouping or independent verification of publication timestamps.

- **Qualcomm:** earlier retained direct regular-session representatives include October 27 at 14:00:24 and 14:20 UTC, before the sampled 14:34:58 item. A saved Reuters-title record also exists at 22:23:34 UTC, much later than the public syndication time. Investigate version/feed timestamp differences; do not assume the saved timestamp is first publication. Other chip-launch coverage was excluded for insufficient NVIDIA connection.
- **Amazon:** earlier retained direct regular-session representatives appear December 2 at 16:07:20, 16:44:32 and 18:31:52 UTC, before the sampled 19:17 item.
- **Microsoft:** an earlier direct retained regular-session record appears January 26 at 16:01:09 UTC, before the sampled 16:03:12 item. The broader collection contains even earlier-clock records, but some are classified indirect or excluded for insufficient NVIDIA connection, and their saved timestamps require validation.
- **OpenAI:** the exact TechCrunch headline used as external evidence is in the collection at February 27 14:13:01 UTC, but excluded for `summary_only_mention_without_material_nvidia_event`. Earlier same-development direct representatives also exist at saved times 08:31:11 and 13:35:10 UTC, outside the default regular-session scope. Some other early stories are excluded by generic-recommendation or summary-only rules. Those exclusion reasons deserve review; presence of a rule label does not prove the exclusion was appropriate. Saved very-early-clock records are not independently verified release timestamps.

The `default_news_eligible` field in the match CSV checks direct relevance, non-exclusion, representative status and regular-session clock on these particular normal trading dates. It does not check quote availability, clean-horizon eligibility or final train/test inclusion.

## Revised interpretation

There is evidence of earlier coverage being present and sometimes retained. The issue may be repeated coverage counted as separate events, exclusion of useful stories, regular-session restriction, and uncertain feed timestamps. It is not established that the collection simply missed the announcements.

A fixed five-minute cluster groups nearby articles; it does not necessarily group all later coverage of the same underlying announcement. The distinct retained records above can therefore form multiple clusters with different outcome windows. To determine effects on the ARC run, trace article IDs through `cluster_articles.parquet` and `news_price_events.parquet`, then examine eligibility and outcomes. Do not infer final training inclusion from the news CSV alone.

Keep the original six-candidate timing report as a record of that sample's external checks, but interpret its public-report anchors as located evidence, not the earliest record in this dataset or the proven first public disclosure.
