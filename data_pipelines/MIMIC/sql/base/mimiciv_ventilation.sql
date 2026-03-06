select 
        v.stay_id,
        extract(epoch from v.starttime) as starttime, 
        extract(epoch from v.endtime) as endtime,
        v.ventilation_status as status
from mimiciv_derived.ventilation v
order by stay_id asc, starttime asc
