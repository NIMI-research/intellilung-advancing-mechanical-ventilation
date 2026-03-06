select 
	A.stay_id, 
        extract(epoch from A.charttime) as charttime,
        A.itemid,
        A.value,
        A.valueuom,
	case
			-- insert itemids and names here
	else B.label
    	end as label,
    	case 
    			-- insert itemids and priorities here
    	end as priority
	from mimiciv_icu.outputevents A
	    INNER JOIN mimiciv_icu.d_items B ON A.itemid = B.itemid
	where  A.value is not null
	    and A.itemid in (
			-- insert just itemids here  
	    )  
	order by stay_id, charttime
