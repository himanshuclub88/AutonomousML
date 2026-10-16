
from __future__ import annotations
import json, re
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split

SUPPORTED_TASKS={"classification","regression","forecasting","clustering","anomaly_detection","dimensionality_reduction"}

def inspect_csv(path: str) -> tuple[pd.DataFrame,list[str]]:
    file_path=Path(path).expanduser().resolve()
    if not file_path.is_file(): raise FileNotFoundError(f"CSV not found: {file_path}")
    df=pd.read_csv(file_path)
    if df.empty or len(df.columns)<2: raise ValueError("CSV must contain rows and at least two columns.")
    if not df.columns.is_unique: raise ValueError("CSV contains duplicate column names.")
    return df,list(df.columns)

def generate_synthetic(problem_type: str,target: str|None,features: list[str],rows: int,*,use_case: str="",random_state: int=42)->pd.DataFrame:
    if problem_type not in SUPPORTED_TASKS: raise ValueError(f"Unsupported problem type: {problem_type!r}")
    if rows<30: raise ValueError("At least 30 rows are required.")
    features=list(dict.fromkeys(c for c in features if c and c!=target))
    if not features: raise ValueError("At least one feature is required.")
    nf=max(2,len(features))
    if problem_type=="classification":
        X,y=make_classification(n_samples=rows,n_features=nf,n_informative=max(2,min(nf,6)),n_redundant=0,n_repeated=0,n_classes=2,random_state=random_state)
    elif problem_type in {"regression","forecasting"}:
        X,y=make_regression(n_samples=rows,n_features=nf,n_informative=min(nf,8),noise=15,random_state=random_state)
        if problem_type=="forecasting": y=y+0.02*np.arange(rows)
    else:
        rng=np.random.default_rng(random_state); X=rng.normal(size=(rows,nf))
        if problem_type=="clustering":
            centers=np.array([np.zeros(nf),np.full(nf,4.0),np.full(nf,-4.0)])
            groups=rng.integers(0,len(centers),size=rows)
            X=centers[groups]+rng.normal(scale=0.8,size=(rows,nf))
        elif problem_type=="anomaly_detection":
            count=max(1,rows//50); X[-count:]+=rng.normal(loc=6.0,scale=1.0,size=(count,nf))
        else:
            latent=rng.normal(size=(rows,min(3,nf))); projection=rng.normal(size=(latent.shape[1],nf))
            X=latent@projection+rng.normal(scale=0.05,size=(rows,nf))
        y=None
    names=list(features)
    while len(names)<nf: names.append(f"feature_{len(names)+1}")
    df=pd.DataFrame(X,columns=names[:nf])
    if target is not None and y is not None: df[target]=y
    return df

def validate_generated_data(df: pd.DataFrame,target: str|None,features: list[str],expected_rows: int)->None:
    required=list(features)+([target] if target else [])
    missing=[c for c in required if c not in df.columns]
    if missing: raise ValueError(f"Missing generated columns: {missing}")
    if len(df)!=expected_rows: raise ValueError(f"Expected {expected_rows} rows; got {len(df)}.")
    if df.empty: raise ValueError("Generated dataset is empty.")
    if df[required].isna().any().any(): raise ValueError("Generated dataset contains missing values.")
    if target and df[target].nunique()<2: raise ValueError("Target must have at least two unique values.")

def split_data(frame: pd.DataFrame,target: str,problem_type: str):
    if target not in frame.columns: raise ValueError(f"Target column {target!r} is not in the dataset.")
    clean=frame.dropna(subset=[target]).copy()
    if len(clean)<30: raise ValueError("Need at least 30 usable rows for training.")
    X,y=clean.drop(columns=[target]),clean[target]
    if X.empty: raise ValueError("No input features remain.")
    if problem_type=="forecasting":
        n=len(clean); a=int(n*.70); b=int(n*.85)
        return X.iloc[:a].copy(),X.iloc[a:b].copy(),X.iloc[b:].copy(),y.iloc[:a].copy(),y.iloc[a:b].copy(),y.iloc[b:].copy()
    stratify=y if problem_type=="classification" and y.nunique()>1 and y.value_counts().min()>=3 else None
    Xtr,Xtmp,ytr,ytmp=train_test_split(X,y,test_size=.30,random_state=42,stratify=stratify)
    st=ytmp if problem_type=="classification" and ytmp.nunique()>1 and ytmp.value_counts().min()>=2 else None
    Xv,Xte,yv,yte=train_test_split(Xtmp,ytmp,test_size=.50,random_state=42,stratify=st)
    return Xtr,Xv,Xte,ytr,yv,yte

def prepare_dataset(frame: pd.DataFrame,target: str|None,features: list[str],problem_type: str,*,remove_duplicates: bool=False,min_rows: int=30)->tuple[pd.DataFrame,dict]:
    if not isinstance(frame,pd.DataFrame): raise TypeError("frame must be a pandas DataFrame.")
    if problem_type not in SUPPORTED_TASKS: raise ValueError(f"Unsupported problem type: {problem_type!r}")
    features=list(dict.fromkeys(features))
    if target in features: features.remove(target)
    if not features: raise ValueError("At least one feature column is required.")
    if problem_type in {"classification","regression","forecasting"} and not target: raise ValueError(f"{problem_type} requires a target column.")
    required=features+([target] if target else [])
    missing=[c for c in required if c not in frame.columns]
    if missing: raise ValueError(f"Planned columns are missing from dataset: {missing}")
    clean=frame[required].copy(); rows_before=len(clean)
    missing_before={str(c):int(n) for c,n in clean.isna().sum().items() if n}
    infinities={}
    for c in clean.select_dtypes(include=[np.number]).columns:
        mask=np.isinf(clean[c].to_numpy(dtype=float,na_value=np.nan))
        if mask.any(): infinities[str(c)]=int(mask.sum()); clean.loc[mask,c]=np.nan
    dropped=0
    if target:
        dropped=int(clean[target].isna().sum()); clean=clean.dropna(subset=[target]).copy()
        if problem_type=="regression" and not pd.api.types.is_numeric_dtype(clean[target]): raise ValueError(f"Regression target {target!r} must be numeric.")
        if clean[target].nunique(dropna=True)<2: raise ValueError("Target has fewer than two unique values.")
    all_missing=[c for c in features if clean[c].isna().all()]
    if all_missing: raise ValueError(f"Features contain no usable values: {all_missing}")
    duplicates=int(clean.duplicated(subset=required).sum()); removed=0
    if remove_duplicates and duplicates:
        n=len(clean); clean=clean.drop_duplicates(subset=required).copy(); removed=n-len(clean)
    if len(clean)<min_rows: raise ValueError(f"Only {len(clean)} usable rows remain; at least {min_rows} are required.")
    report={"status":"passed","problem_type":problem_type,"target_column":target,"feature_columns":features,"rows_before":int(rows_before),"rows_after":int(len(clean)),"rows_dropped_for_missing_target":dropped,"duplicate_rows_detected":duplicates,"duplicate_rows_removed":removed,"remove_duplicates_enabled":remove_duplicates,"missing_values_before":missing_before,"infinite_values_converted_to_missing":infinities,"missing_values_after":{str(c):int(n) for c,n in clean[required].isna().sum().items() if n},"constant_features":[c for c in features if clean[c].nunique(dropna=True)<=1],"all_missing_features":all_missing,"target_unique_values":int(clean[target].nunique()) if target else None,"target_class_counts":{str(k):int(v) for k,v in clean[target].value_counts().items()} if target and problem_type=="classification" else None,"notes":["Feature missing values are handled by training imputers.","Duplicates are preserved unless removal is enabled.","Quality checks do not prove data is unbiased or leakage-free."]}
    return clean,report

def write_data_quality_report(report: dict,path: str|Path)->None:
    output=Path(path); output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf-8")

def analyze_features(frame: pd.DataFrame,target: str|None,features: list[str],*,high_cardinality_ratio: float=.50,identifier_unique_ratio: float=.90)->dict:
    if target is not None and target not in frame.columns: raise ValueError(f"Target column {target!r} is not present.")
    missing=[c for c in features if c not in frame.columns]
    if missing: raise ValueError(f"Feature columns are missing: {missing}")
    idpat=re.compile(r"(^|[\\s_\\-])(id|uuid|guid|key|index|identifier|record|account|customer_id|transaction_id|row_number)([\\s_\\-]|$)",re.I)
    leakpat=re.compile(r"(target|label|outcome|actual|ground.?truth|future|post.?event|final.?price|sale.?price|sold.?price|price.?after)",re.I)
    reports=[]
    for c in features:
        s=frame[c]; values=s.dropna(); n=int(values.nunique()); ratio=float(n/max(1,len(values))); flags=[]; rec=[]
        if n<=1: flags.append("constant_feature"); rec.append("Consider excluding: no variation.")
        if idpat.search(str(c)): flags.append("identifier_like_name"); rec.append("Review whether this is an identifier.")
        cat=pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or isinstance(s.dtype,pd.CategoricalDtype)
        if cat and ratio>=identifier_unique_ratio and len(values)>=20: flags.append("near_unique_values"); rec.append("Review for unique identifiers or non-generalizing categories.")
        if cat and n>20 and ratio>=high_cardinality_ratio: flags.append("high_cardinality_categorical"); rec.append("Review encoding cost and generalization.")
        if leakpat.search(str(c)): flags.append("possible_target_leakage_by_name"); rec.append("Check target leakage and prediction-time availability.")
        reports.append({"column":str(c),"dtype":str(s.dtype),"rows":int(len(s)),"missing_values":int(s.isna().sum()),"unique_values":n,"unique_ratio":round(ratio,4),"flags":flags,"recommendations":rec})
    return {"target_column":target,"feature_count":len(features),"features_flagged":sum(bool(x["flags"]) for x in reports),"feature_analysis":reports,"important_limitations":["Heuristic warnings are not proof of harm or leakage.","Columns are not automatically removed.","High-cardinality address features may overfit or increase encoding cost."]}
