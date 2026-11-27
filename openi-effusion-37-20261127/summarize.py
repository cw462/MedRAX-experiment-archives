"""Summarize the two matched cohorts from the archived observable events."""
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

root = Path(__file__).resolve().parent
questions = [json.loads(p.read_bytes()) for cohort in ('first18','new19') for p in sorted((root/cohort).glob('*.json'))]
conditions = ('original','heatmap','joint','xraydar')
records, totals = [], []
events = defaultdict(list)
for q in questions:
    for model in ('qwen','deepseek'):
        group = {r['condition']:r for r in q['runs'] if r['model']==model and r['batch'] in ('paired18','paired19')}
        for cond in conditions:
            r = group[cond]
            base = group['original']['score']['correct']
            ok = r['score']['correct']
            tools = [t.get('tool') or '' for t in r['tool_events']]
            record = dict(cohort=q['cohort'],ordinal=r['ordinal'],qid=q['question_id'],model=model,condition=cond,reference=q['reference'],choice=r['score']['choice'],correct=int(ok),corrected=int(ok and not base),regressed=int(base and not ok),main_calls=r['score']['main_calls'],tool_calls=len(tools),failed_tools=r['score']['failed_tool_calls'],memory_items=r['memory_before']['items'],memory_chars=r['memory_before']['memory_chars'],tool_sequence=' > '.join(tools),file=q['cohort']+'/'+q['question_id']+'.json')
            records.append(record)
            events[(q['cohort'],model,cond)].append(record)
for cohort in ('first18','new19','both'):
    for model in ('qwen','deepseek'):
        for cond in conditions:
            rows=[r for r in records if r['model']==model and r['condition']==cond and (cohort=='both' or r['cohort']==cohort)]
            total={k:sum(r[k] for r in rows) for k in ('correct','corrected','regressed','main_calls','tool_calls','failed_tools','memory_items')}
            totals.append(dict(cohort=cohort,model=model,condition=cond,n=len(rows),net=total['corrected']-total['regressed'],with_tools=sum(r['tool_calls']>0 for r in rows),with_memory=sum(r['memory_items']>0 for r in rows),single_decision=sum(r['main_calls']==1 for r in rows),**total))
for file,rows in [('question_index.csv',records),('trajectory_summary.csv',totals)]:
    with (root/file).open('w',newline='',encoding='utf-8-sig') as handle:
        writer=csv.DictWriter(handle,fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
