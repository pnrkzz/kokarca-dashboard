from __future__ import annotations
from functools import lru_cache
from pathlib import Path
from collections import Counter
import os
import pandas as pd
from fastapi import APIRouter, HTTPException, Query

router = APIRouter(prefix='/api/v22/interventions', tags=['V22 interventions'])
ROOT = Path(os.getenv('KOKARCA_V22_ROOT', r'C:\Users\depoc\kokarca_project\data_processed\date_audit\v14_final_validation'))
LOCK = ROOT / 'v22_final_architecture_lock'
MASTER = LOCK / 'hhalys_final_validated_event_master_3769_V22_LOCKED.csv'
FACTS = LOCK / 'v22_intervention_long.csv'
ACTORS = LOCK / 'v22_actor_long.csv'

def is_true(s):
    return s.fillna('').astype(str).str.strip().str.lower().isin(['true', '1', 'yes'])

@lru_cache(maxsize=1)
def load():
    for p in [MASTER, FACTS, ACTORS]:
        if not p.is_file():
            raise FileNotFoundError(str(p))
    master = pd.read_csv(MASTER, dtype=str, low_memory=False).fillna('')
    facts = pd.read_csv(FACTS, dtype=str, low_memory=False).fillna('')
    actors = pd.read_csv(ACTORS, dtype=str, low_memory=False).fillna('')
    keys = ['global_id', 'event_order']
    if master.duplicated(keys).any():
        raise ValueError('V22 master has duplicate event keys')
    master_fields = keys + ['source_type', 'analysis_year_final', 'eligible_yearly_analysis', 'final_event_role', 'crop_host_names_final', 'event_summary']
    for c in master_fields:
        if c not in master.columns:
            master[c] = ''
    merged = facts.merge(master[master_fields], on=keys, how='left', validate='many_to_one', indicator='_master_match')
    merged['master_matched'] = merged['_master_match'].eq('both')
    merged['supported'] = is_true(merged['intervention_supported'])
    merged['year_eligible'] = is_true(merged['eligible_yearly_analysis'])
    merged['year'] = pd.to_numeric(merged['analysis_year_final'], errors='coerce')
    merged.loc[~merged['year_eligible'], 'year'] = float('nan')
    actors = actors[actors['global_id'].notna() & actors['event_order'].notna()]
    actor_map = actors.groupby(keys)['actor_text'].apply(lambda v: list(dict.fromkeys(str(x).strip() for x in v if str(x).strip()))).to_dict()
    merged['actors'] = [actor_map.get((str(g), str(e)), []) for g, e in zip(merged.global_id, merged.event_order)]
    return merged

def filtered(supported_only=True, status=None, intervention_type=None, year=None, source_type=None):
    df = load()
    if supported_only:
        df = df[df['supported'] & df['master_matched']]
    if status:
        df = df[df.status == status]
    if intervention_type:
        df = df[df.intervention_type == intervention_type]
    if source_type:
        df = df[df.source_type == source_type]
    if year is not None:
        df = df[df.year == year]
    return df

def counts(df, col):
    return [{'name': str(k), 'count': int(v)} for k, v in df[col].fillna('Belirsiz').replace('', 'Belirsiz').value_counts().items()]

@router.get('/summary')
def summary():
    try:
        all_df = load()
        df = filtered()
        dated = df[df.year.notna() & df.year.between(2015, 2026)]
        years = dated.groupby('year').size().sort_index()
        return {
            'data_version': 'V22_LOCKED', 'unit': 'intervention_fact',
            'total_fact_rows': int(len(all_df)),
            'master_matched_rows': int(all_df.master_matched.sum()),
            'supported_matched_rows': int(len(df)),
            'unique_events': int(df[['global_id', 'event_order']].drop_duplicates().shape[0]),
            'type_counts': counts(df, 'intervention_type'),
            'status_counts': counts(df, 'status'),
            'source_counts': counts(df, 'source_type'),
            'year_counts': [{'year': int(y), 'count': int(n)} for y, n in years.items()],
            'geographic_scope': 'national_only_unverified_fact_location',
            'caution': 'Supported means extraction evidence supported, not verified field implementation. Status is source-classified. Year is event-level and not independently fact-dated. Province is intentionally not attributed.'
        }
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=str(exc))

@router.get('/records')
def records(limit: int = Query(40, ge=1, le=200), offset: int = Query(0, ge=0), status: str | None = None, intervention_type: str | None = None, year: int | None = None, source_type: str | None = None):
    try:
        df = filtered(status=status, intervention_type=intervention_type, year=year, source_type=source_type)
        cols = ['global_id', 'event_order', 'fact_order', 'source_type', 'source_stage', 'intervention_type', 'status', 'fact_text', 'intervention_method', 'reported_outcome', 'year', 'actors']
        page = df.iloc[offset:offset+limit][cols].copy()
        page['year'] = pd.to_numeric(page['year'], errors='coerce').apply(lambda v: int(v) if pd.notna(v) else None)
        return {'total': int(len(df)), 'records': page.astype(object).where(pd.notna(page), None).to_dict('records'), 'limit': limit, 'offset': offset}
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=str(exc))
