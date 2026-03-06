select 
        i.stay_id,
        extract(
        	epoch from 
        	i.intime
        ) as intime,
        extract(
        	epoch from 
        	i.outtime
        ) as outtime,
        i.los
        from mimiciv_icu.icustays as i
