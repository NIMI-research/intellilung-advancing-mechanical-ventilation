import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime
import ast

VARID = 'variableid'
PID = 'patientid'
VALUE = 'value'
DATETIME = 'datetime'
ENTERTIME = 'entertime'

REL_DATETIME = 'rel_datetime'

VARREF_LOWERBOUND = 'lowerbound'
VARREF_UPPERBOUND = 'upperbound'

def drop_out_of_range_values(df, varref):
    """
    df: long-format dataframe of a patient
    varref: variable reference table that contain t he lower-bound and upper-bound of values for a subset of variables
    """
    bound_cols = [VARREF_LOWERBOUND, VARREF_UPPERBOUND]

    df_with_bounds = df.merge(varref[bound_cols], left_on=VARID, right_index=True, how='inner')
    df_filtered = df_with_bounds.query(f'{VARREF_LOWERBOUND}.isnull() or {VARREF_UPPERBOUND}.isnull() or ({VARREF_LOWERBOUND} <= value and value <= {VARREF_UPPERBOUND})', engine='python')

    return df_filtered.drop(columns=bound_cols)

def transform_mon_table_fn(monv: pd.DataFrame, varref):
    monv = monv[~monv[VALUE].isna()].sort_values([VARID, DATETIME, ENTERTIME])

    monv = drop_out_of_range_values(monv, varref)

    monv.loc[:, VARID] = monv[VARID].apply(lambda x: "v%d" % x)

    if monv.empty:
        return pd.DataFrame()

    wide_mon = (pd.pivot_table(monv, values=VALUE, columns=VARID, index=DATETIME).
                        sort_index())
    
    convfactor_variables = set(varref[~varref.unitconversionfactor.isna()].index)
    for col in convfactor_variables:
        if "v%d" % col in wide_mon.columns:
            wide_mon.loc[:,"v%d" % col] = wide_mon["v%d" % col] * float(varref.loc[col].unitconversionfactor)
    
    mapping_variables = set(varref[~varref.mapping.isna()].index)
    for col in mapping_variables:
        if "v%d" % col in wide_mon.columns:
            d = ast.literal_eval(varref.loc[col].mapping)
            wide_mon.loc[:, "v%d" % col] = wide_mon["v%d" % col].map(d)

    return wide_mon

def process_single_infusion(df, infusionid):
    '''
    Convert given dose from a single infusion channel to rate
    ''' 
    #Process bolus and coninuous infusions differently
    if df.iloc[0].started == 1 and df.iloc[0].stoped == 1:
        #bolus injection
        if len(df.index) > 1:
            return pd.DataFrame(), True
        df.loc[:, str(infusionid)] = df.iloc[0].givendose
        df_tmp = df.iloc[0][["givenat", str(infusionid)]].to_frame().T
        return df_tmp.set_index("givenat"), True
    else:
        #continuous infusion calculate dose-rate/min
        df.loc[:, str(infusionid)] = 0
        #claculate rate as givendose / past time in minutes and store it with the previous entry (when the rate change actually was made)       
        df.loc[df.index[:-1], str(infusionid)] = df.givendose.values[1:] / (df.givenat.diff() / np.timedelta64(1,"m")).values[1:]
        #delete if two entries have exactly the same time
        same_index = np.append((df.loc[df.index[:-1]].givenat.values == df.loc[df.index[1:]].givenat.values), False)
        df = df.loc[~same_index]
        #end infusion with zero rate (CAVE: ffilling later)
        df.loc[df.index[-1], str(infusionid)] = 0
    
        return df[["givenat", str(infusionid)]].set_index("givenat"), False

def transform_pharma_table_fn(pharma: pd.DataFrame, pharmaref):
    pharma.sort_values(["infusionid", "givenat"], inplace=True)

    if pharma.empty:
        return pd.DataFrame()
    else:
        wide_pharma = []
        for pharmaid in pharma.pharmaid.unique():
            #loop trough each pharmaid sperately
            infusion_dose = []
            bolus_dose = []
            for infusionid in pharma[pharma.pharmaid == pharmaid].infusionid.unique():
                #loop trough each infusion channel              
                tmp_pharma = pharma[(pharma.pharmaid == pharmaid) & (pharma.infusionid == infusionid)].copy().sort_values("givenat")
                flow_series, bolus = process_single_infusion(tmp_pharma, infusionid)
                if bolus:
                    bolus_dose.append(flow_series)
                else:
                    infusion_dose.append(flow_series)

            if len(infusion_dose) > 0:
                infusion_dose = pd.concat(infusion_dose, axis=1).sort_index().fillna(method='ffill')
                infusion_dose = infusion_dose.sum(axis=1).to_frame(name="v%d" % pharmaid)
                wide_pharma.append(infusion_dose)

            if len(bolus_dose) > 0:
                bolus_dose = pd.concat(bolus_dose, axis=1).sort_index()
                bolus_dose = bolus_dose.sum(axis=1).to_frame(name="v%d_bolus" % pharmaid)
                wide_pharma.append(bolus_dose)

        wide_pharma = pd.concat(wide_pharma, axis=1).sort_index()
        
        #correct the unit of drugs if required
        convfactor_variables = set(pharmaref[~pharmaref.unitconversionfactor.isna()].index)
        for col in convfactor_variables:
            if "v%d" % col in wide_pharma.columns:
                wide_pharma.loc[:,"v%d" % col] = wide_pharma["v%d" % col] * float(pharmaref[pharmaref.variableid == col].iloc[0].unitconversionfactor)
                wide_pharma.loc[:,"v%d_bolus" % col] = wide_pharma["v%d_bolus" % col] * float(pharmaref[pharmaref.variableid == col].iloc[0].unitconversionfactor)
        
        #merge into meta variables
        for pmid in pharmaref.metavariableid.unique():
            #for continuous
            cols = ['v%d' % x for x in pharmaref[pharmaref.metavariableid == pmid].index]
            wide_pharma.loc[:, list(set(cols).difference(set(wide_pharma.columns)))] = np.nan
            if np.isin(wide_pharma.columns, cols).sum() == 0:
                wide_pharma.loc[:, "vm%d" % pmid] = np.nan
            else:
                wide_pharma.loc[:, "vm%d" % pmid] = wide_pharma[wide_pharma.columns[np.isin(wide_pharma.columns, cols)]].sum(axis=1)
                wide_pharma.loc[wide_pharma.index[wide_pharma[wide_pharma.columns[np.isin(wide_pharma.columns, cols)]].notnull().sum(axis=1) == 0], "vm%d" % pmid] = np.nan
                wide_pharma.drop(wide_pharma.columns[np.isin(wide_pharma.columns, cols)], axis=1, inplace=True)
            
            #for bolus
            cols = ['v%d_bolus' % x for x in pharmaref[pharmaref.metavariableid == pmid].index]
            wide_pharma.loc[:, list(set(cols).difference(set(wide_pharma.columns)))] = np.nan
            if np.isin(wide_pharma.columns, cols).sum() == 0:
                wide_pharma.loc[:, "vm%d_bolus" % pmid] = np.nan
            else:
                wide_pharma.loc[:, "vm%d_bolus" % pmid] = wide_pharma[wide_pharma.columns[np.isin(wide_pharma.columns, cols)]].sum(axis=1)
                wide_pharma.loc[wide_pharma.index[wide_pharma[wide_pharma.columns[np.isin(wide_pharma.columns, cols)]].notnull().sum(axis=1) == 0], "vm%d_bolus" % pmid] = np.nan
                wide_pharma.drop(wide_pharma.columns[np.isin(wide_pharma.columns, cols)], axis=1, inplace=True)

        #ffill only cont pharma variables
        cont_cols = ['vm%d' % x for x in pharmaref.metavariableid.unique()]
        wide_pharma[cont_cols] = wide_pharma[cont_cols].fillna(method='ffill').fillna(0)
        return wide_pharma
    
def aggregate_cols(wide_observ, varref):
    metavar_varid_dict = {m: [f"v{vid}" for vid in vars] for (m, vars) in varref.reset_index().groupby('metavariableid')['variableid']}

    metavar_cols = {}
    for vmid, varid_cols in metavar_varid_dict.items():
        varid_cols_avail = list(set(varid_cols).intersection(set(wide_observ.columns)))

        if len(varid_cols_avail) > 1:
            c = wide_observ.loc[:, varid_cols_avail].bfill(axis=1).iloc[:, 0]   #takes the value of the first column
        elif len(varid_cols_avail) == 1:
            c = wide_observ.loc[:, varid_cols_avail[0]]
        else:
            c = np.zeros(wide_observ.shape[0])
            c[:] = np.nan

        metavar_cols[f"vm{vmid}"] = c

    wide_observ_new = pd.DataFrame(metavar_cols, index=wide_observ.index)

    return wide_observ_new


def process_data_patient(pid, df_observ, df_drugs, varref):
    print(pid)
    observ_merged = transform_mon_table_fn(df_observ, varref[varref.type == "observed"])
    pharma_merged = transform_pharma_table_fn(df_drugs, varref[varref.type == "pharma"])

    if not observ_merged.empty:
        observ_merged = observ_merged.reset_index().rename(columns={DATETIME: "AbsDatetime", "index": "AbsDatetime"}).resample('5min', on='AbsDatetime').last()
        observ_merged = aggregate_cols(observ_merged, varref[varref.type != "pharma"]).reset_index()
    
    if not pharma_merged.empty:
        pharma_merged = pharma_merged.reset_index().rename(columns={"givenat": "AbsDatetime", "index": "AbsDatetime"}).resample('5min', on='AbsDatetime').last()
        df_pid = observ_merged.join(pharma_merged, on="AbsDatetime", how="outer")
    else:
        df_pid = observ_merged
    
    df_pid["PatientID"] = pid 
    
    return df_pid