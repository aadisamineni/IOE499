"""Word counts for equal-hour windows around April's top ten moves."""
import sys,html,re,json
from pathlib import Path
from collections import Counter
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from News_word_frequency.group_word_frequency import STOPWORDS
OUT=Path(__file__).resolve().parent

def main():
    pairs=pd.read_csv(OUT/'event_article_matches.csv',keep_default_na=False)
    pairs=pairs[pairs['rank'].astype(int).le(10)]
    results=[];groups=[];audit=[]
    for group,flag in [('before','before_1h'),('during','during'),('after','after_1h')]:
        selected=pairs[pairs[flag].astype(str).str.lower().eq('true')]
        articles=selected.drop_duplicates('article_id')
        counts=Counter();docs=Counter()
        for _,a in articles.iterrows():
            text=html.unescape(a.headline+' '+a.summary).lower().replace('’',"'")
            text=re.sub(r'https?://\S+|www\.\S+',' ',text)
            text=re.sub(r"\b([a-z]+)'s\b",r'\1',text)
            words=[w for w in re.findall(r'[a-z]+',text) if len(w)>1 and w not in STOPWORDS]
            counts.update(words);docs.update(set(words))
            audit.append({'group':group,'article_id':a.article_id,'headline':a.headline,'published_at_utc':a.published_at_utc})
        groups.append({'group':group,'event_article_pairs':len(selected),'unique_articles':len(articles),'retained_tokens':sum(counts.values())})
        for rank,(word,count) in enumerate(sorted(counts.items(),key=lambda x:(-x[1],x[0])),1):
            results.append({'group':group,'rank':rank,'word':word,'occurrences':count,'articles_containing_word':docs[word],'group_unique_articles':len(articles),'article_prevalence_pct':100*docs[word]/len(articles)})
    full=pd.DataFrame(results);top=full[full['rank'].le(5)]
    full.to_csv(OUT/'word_frequencies.csv',index=False);top.to_csv(OUT/'top5_words.csv',index=False)
    pd.DataFrame(audit).to_csv(OUT/'word_group_articles.csv',index=False)
    (OUT/'word_count_settings.json').write_text(json.dumps({'groups':groups,'stopwords':sorted(STOPWORDS),'ranking':'Total token occurrences; alphabetical ties','text':'HTML-decoded headline + summary; lowercase; possessive suffix removed; alphabetic words length >1; no stemming','scope':'Top 10 moves, equal 1h before/during/after; each article once per group; groups may overlap'},indent=2))
    print(pd.DataFrame(groups).to_string(index=False));print(top.to_string(index=False))
if __name__=='__main__':main()
