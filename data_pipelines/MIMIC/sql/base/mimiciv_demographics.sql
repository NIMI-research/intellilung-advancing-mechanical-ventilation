select 
        ad.subject_id, ad.hadm_id, i.stay_id,
        p.anchor_age + EXTRACT(EPOCH FROM ad.admittime - TO_TIMESTAMP(TO_CHAR(p.anchor_year, '0000') || TO_CHAR(1, '00') || TO_CHAR(1, '00') || TO_CHAR(0, '00') || TO_CHAR(0, '00') || TO_CHAR(0, '00'), 'yyyymmddHH24MISS')) / 31556908.8 AS age,
        extract(epoch from ad.admittime) as admittime, 
        extract(epoch from ad.dischtime) as dischtime, 
        ROW_NUMBER() over (partition by ad.subject_id order by i.intime asc) as adm_order,
        case 
            when i.first_careunit='Medical/Surgical Intensive Care Unit (MICU/SICU)' then 5 
            when i.first_careunit='Surgical Intensive Care Unit (SICU)' then 2 
            when i.first_careunit='Cardiac Vascular Intensive Care Unit (CVICU)' then 4 
            when i.first_careunit='Coronary Care Unit (CCU)' then 6 
            when i.first_careunit='Medical Intensive Care Unit (MICU)' then 1 
            when i.first_careunit='Trauma SICU (TSICU)' then 3 
            when i.first_careunit='Neuro Intermediate' then 7
            when i.first_careunit='Neuro Stepdown' then 8
            when i.first_careunit='Neuro Surgical Intensive Care Unit (Neuro SICU)' then 9
        end as unit,  
        extract(epoch from i.intime) as intime, 
        extract(epoch from i.outtime) as outtime, 
        i.los,
        extract(epoch from p.dod) as dod,
        case 
            when p.gender='M' then 0 
            when p.gender='F' then 1 
        end as gender,
        CAST(extract(epoch from age(p.dod,ad.dischtime))<=24*3600 as int) as morta_hosp,  --died in hosp if recorded DOD is close to hosp discharge
        CAST(extract(epoch from age(p.dod,i.intime))<=90*24*3600 as int) as morta_90
    from 
        mimiciv_hosp.admissions ad, 
        mimiciv_icu.icustays i, 
        mimiciv_hosp.patients p
    where 
        ad.hadm_id=i.hadm_id and p.subject_id=i.subject_id
        order by subject_id asc, intime asc
