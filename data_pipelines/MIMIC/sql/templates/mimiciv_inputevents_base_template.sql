select 
        A.stay_id, 
        extract(epoch from A.starttime) as charttime,
        extract(epoch from A.endtime) as endtime,
        A.itemid,
        A.amount as value,
        A.amountuom as valueuom,
	case
			-- insert itemids and names here
	else B.label
    	end as label,
    	case 
    			-- insert itemids and priorities here
    	end as priority
	from mimiciv_icu.inputevents A
	    INNER JOIN mimiciv_icu.d_items B ON A.itemid = B.itemid
	where  A.amount is not null
	    and A.itemid in (
			-- insert just itemids here  
	    )  
	order by stay_id, charttime
