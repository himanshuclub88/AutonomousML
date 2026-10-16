from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import json, time
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score, silhouette_score, davies_bouldin_score, calinski_harabasz_score, roc_auc_score, average_precision_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor, IsolationForest
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor

MODEL_ALLOWLIST = {
 "classification": {"logistic_regression","random_forest","extra_trees","hist_gradient_boosting"},
 "regression": {"linear_regression","random_forest","extra_trees","hist_gradient_boosting"},
 "forecasting": {"linear_regression","random_forest","extra_trees"},
 "clustering": {"kmeans","dbscan","agglomerative_clustering"},
 "anomaly_detection": {"isolation_forest","one_class_svm","local_outlier_factor"},
 "dimensionality_reduction": {"pca","truncated_svd"},
}

def _build_estimator(name: str, problem_type: str):
    maps = {
      "classification": {
        "logistic_regression": LogisticRegression(max_iter=1000),
        "random_forest": RandomForestClassifier(n_estimators=150,random_state=42,n_jobs=1),
        "extra_trees": ExtraTreesClassifier(n_estimators=150,random_state=42,n_jobs=1),
        "hist_gradient_boosting": HistGradientBoostingClassifier(random_state=42)},
      "regression": {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(n_estimators=150,random_state=42,n_jobs=1),
        "extra_trees": ExtraTreesRegressor(n_estimators=150,random_state=42,n_jobs=1),
        "hist_gradient_boosting": HistGradientBoostingRegressor(random_state=42)},
      "forecasting": {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(n_estimators=150,random_state=42,n_jobs=1),
        "extra_trees": ExtraTreesRegressor(n_estimators=150,random_state=42,n_jobs=1)},
      "clustering": {"kmeans": KMeans(n_clusters=3,random_state=42,n_init=10),
        "dbscan": DBSCAN(), "agglomerative_clustering": AgglomerativeClustering(n_clusters=3)},
      "anomaly_detection": {"isolation_forest": IsolationForest(random_state=42),
        "one_class_svm": OneClassSVM(), "local_outlier_factor": LocalOutlierFactor(novelty=True)},
      "dimensionality_reduction": {"pca": PCA(n_components=0.95,random_state=42),
        "truncated_svd": TruncatedSVD(n_components=1,random_state=42)}
    }
    if problem_type not in maps or name not in maps[problem_type]:
        raise ValueError(f"Model {name!r} is not supported for {problem_type!r}.")
    return maps[problem_type][name]

def _pipeline(frame, estimator):
    nums = frame.select_dtypes(include=["number","bool"]).columns.tolist()
    cats = [c for c in frame.columns if c not in nums]
    transformers = []
    if nums:
        transformers.append(("numeric",Pipeline([("imputer",SimpleImputer(strategy="median")),("scaler",StandardScaler())]),nums))
    if cats:
        transformers.append(("categorical",Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore",sparse_output=False))]),cats))
    if not transformers:
        raise ValueError("No usable features.")
    return Pipeline([("preprocess",ColumnTransformer(transformers=transformers)),("model",estimator)])

def _score_values(model, X):
    if hasattr(model,"predict_proba"):
        p=model.predict_proba(X)
        if p.ndim==2 and p.shape[1]>1: return p[:,1]
    if hasattr(model,"decision_function"):
        s=model.decision_function(X)
        if np.ndim(s)==1: return s
    return None

def _metric(task, metric, y, pred, scores=None):
    if task in {"regression","forecasting"}:
        if metric=="mae": return float(mean_absolute_error(y,pred))
        if metric=="r2": return float(r2_score(y,pred))
        if metric=="mape":
            return float(np.mean(np.abs((np.asarray(y)-pred)/np.maximum(np.abs(np.asarray(y)),1e-8)))*100)
        if metric=="smape":
            return float(np.mean(2*np.abs(np.asarray(y)-pred)/np.maximum(np.abs(np.asarray(y))+np.abs(pred),1e-8))*100)
        return float(np.sqrt(mean_squared_error(y,pred)))
    if task=="classification":
        if metric=="accuracy": return float(accuracy_score(y,pred))
        if metric=="roc_auc":
            if scores is None: raise ValueError("ROC AUC requires probability/decision scores.")
            return float(roc_auc_score(y,scores))
        if metric=="average_precision":
            if scores is None: raise ValueError("Average precision requires probability/decision scores.")
            return float(average_precision_score(y,scores))
        return float(f1_score(y,pred,average="weighted",zero_division=0))
    raise ValueError(f"Unsupported supervised metric: {metric}")

def _better(task, metric, a, b):
    if task in {"regression","forecasting"} and metric in {"rmse","mae","mape","smape"}: return a < b
    if task=="clustering" and metric=="davies_bouldin_score": return a < b
    return a > b

def _supervised(frame,target,features,task,candidates,metric,run_dir,workers):
    from .dataset import split_data
    frame=frame[features+[target]].copy()
    Xtr,Xv,Xte,ytr,yv,yte=split_data(frame,target,task)
    results=[]; errors=[]
    def run_one(name):
        t=time.time(); pipe=_pipeline(Xtr,_build_estimator(name,task)); pipe.fit(Xtr,ytr)
        pred=pipe.predict(Xv); score=_metric(task,metric,yv,pred,_score_values(pipe,Xv))
        path=run_dir/f"{name}.joblib"; joblib.dump(pipe,path)
        return {"model":name,"validation_metric":metric,"validation_score":score,"duration_seconds":round(time.time()-t,2),"artifact":str(path)}
    with ThreadPoolExecutor(max_workers=max(1,min(workers,len(candidates)))) as pool:
        fs={pool.submit(run_one,n):n for n in candidates}
        for f in as_completed(fs):
            try: results.append(f.result())
            except Exception as e: errors.append({"model":fs[f],"error":f"{type(e).__name__}: {e}"})
    if not results: raise RuntimeError(f"All model experiments failed: {errors}")
    best=results[0]
    for r in results[1:]:
        if _better(task,metric,r["validation_score"],best["validation_score"]): best=r
    Xtv=pd.concat([Xtr,Xv]); ytv=pd.concat([ytr,yv])
    final=_pipeline(Xtv,_build_estimator(best["model"],task)); final.fit(Xtv,ytv)
    test_score=_metric(task,metric,yte,final.predict(Xte),_score_values(final,Xte))
    path=run_dir/"best_model.joblib"; joblib.dump(final,path)
    return {"problem_type":task,"target_column":target,"features":features,"selection_metric":metric,
      "best_model":best["model"],"best_validation_score":best["validation_score"],"held_out_test_score":test_score,
      "test_metric":metric,"experiments":sorted(results,key=lambda r:r["duration_seconds"]),"errors":errors,
      "best_model_path":str(path),"rows_used":len(frame)}

def _unsupervised(frame,features,task,candidates,metric,run_dir,workers):
    X=frame[features].copy(); results=[]; errors=[]
    def run_one(name):
        t=time.time(); pipe=_pipeline(X,_build_estimator(name,task)); pipe.fit(X)
        est=pipe.named_steps["model"]; Xt=pipe.named_steps["preprocess"].transform(X)
        score=None; score_metric="not_applicable"
        if task=="clustering":
            labels=getattr(est,"labels_",None)
            if labels is None and hasattr(est,"predict"): labels=est.predict(Xt)
            if labels is not None:
                labels=np.asarray(labels); valid=labels!=-1; unique=np.unique(labels[valid])
                if len(unique)>=2 and valid.sum()>len(unique):
                    if metric=="silhouette_score": score=float(silhouette_score(Xt[valid],labels[valid])); score_metric=metric
                    elif metric=="davies_bouldin_score": score=float(davies_bouldin_score(Xt[valid],labels[valid])); score_metric=metric
                    elif metric=="calinski_harabasz_score": score=float(calinski_harabasz_score(Xt[valid],labels[valid])); score_metric=metric
        elif task=="anomaly_detection" and metric in {"roc_auc","average_precision"}:
            raise ValueError(f"{metric} requires labelled anomalies; use not_applicable without labels.")
        elif task=="dimensionality_reduction" and hasattr(est,"explained_variance_ratio_"):
            score=float(np.sum(est.explained_variance_ratio_)); score_metric="explained_variance"
        path=run_dir/f"{name}.joblib"; joblib.dump(pipe,path)
        return {"model":name,"validation_metric":score_metric,"validation_score":score,"duration_seconds":round(time.time()-t,2),"artifact":str(path)}
    with ThreadPoolExecutor(max_workers=max(1,min(workers,len(candidates)))) as pool:
        fs={pool.submit(run_one,n):n for n in candidates}
        for f in as_completed(fs):
            try: results.append(f.result())
            except Exception as e: errors.append({"model":fs[f],"error":f"{type(e).__name__}: {e}"})
    if not results: raise RuntimeError(f"All model experiments failed: {errors}")
    scored=[r for r in results if isinstance(r["validation_score"],(int,float)) and np.isfinite(r["validation_score"])]
    best=scored[0] if scored and metric!="not_applicable" else results[0]
    for r in (scored[1:] if scored and metric!="not_applicable" else []):
        if _better(task,metric,r["validation_score"],best["validation_score"]): best=r
    path=run_dir/"best_model.joblib"; joblib.dump(joblib.load(best["artifact"]),path)
    return {"problem_type":task,"target_column":None,"features":features,"selection_metric":metric,
      "best_model":best["model"],"best_validation_score":best["validation_score"],"held_out_test_score":None,
      "test_metric":None,"experiments":sorted(results,key=lambda r:r["duration_seconds"]),"errors":errors,
      "best_model_path":str(path),"rows_used":len(frame),"note":"Unsupervised diagnostics are not held-out predictive performance."}

def train_candidates(frame: pd.DataFrame,target: str|None,features: list[str],problem_type: str,
                     candidates: list[str],metric: str,run_dir: Path,workers: int=3) -> dict:
    run_dir=Path(run_dir); run_dir.mkdir(parents=True,exist_ok=True)
    if not isinstance(frame,pd.DataFrame) or frame.empty: raise ValueError("frame must be a non-empty DataFrame.")
    if problem_type not in MODEL_ALLOWLIST: raise ValueError(f"Unsupported problem type: {problem_type!r}")
    if not candidates: raise ValueError("At least one candidate model is required.")
    features=list(dict.fromkeys(features))
    if not features: raise ValueError("At least one feature column is required.")
    missing=[c for c in features if c not in frame.columns]
    if missing: raise ValueError(f"Missing feature columns: {missing}")
    for name in candidates:
        if name not in MODEL_ALLOWLIST[problem_type]: raise ValueError(f"Model {name!r} is not supported for {problem_type!r}.")
    if problem_type in {"classification","regression","forecasting"}:
        if not target or target not in frame.columns: raise ValueError(f"{problem_type} requires a target column.")
        if target in features: raise ValueError("Target column must not be in features.")
        result=_supervised(frame,target,features,problem_type,candidates,metric,run_dir,workers)
    else:
        result=_unsupervised(frame,features,problem_type,candidates,metric,run_dir,workers)
    (run_dir/"metrics.json").write_text(json.dumps(result,indent=2,default=str),encoding="utf-8")
    _write_training_script(run_dir/"train_model.py",target,features,problem_type,result["best_model"],metric)
    return result

def _split(frame,target,problem_type):
    from .dataset import split_data
    return split_data(frame,target,problem_type)

def _write_training_script(path: Path,target: str|None,features: list[str],problem_type: str,model: str,metric: str):
    # Generated script loads the winning artifact and actually tests its output.
    lines = [
      "# Generated by Autonomous ML Engineering Agent.",
      "from pathlib import Path",
      "import json, joblib, pandas as pd",
      "CSV_PATH = Path('your_dataset.csv')",
      "MODEL_PATH = Path(__file__).with_name('best_model.joblib')",
      f"FEATURES = {features!r}", f"TARGET = {target!r}",
      f"PROBLEM_TYPE = {problem_type!r}", f"MODEL_NAME = {model!r}", f"METRIC = {metric!r}",
      "def main():",
      "    if not CSV_PATH.is_file(): raise FileNotFoundError(f'Set CSV_PATH to a real CSV: {CSV_PATH}')",
      "    df = pd.read_csv(CSV_PATH)",
      "    missing = [c for c in FEATURES if c not in df.columns]",
      "    if missing: raise ValueError(f'Missing features: {missing}')",
      "    model = joblib.load(MODEL_PATH); X = df[FEATURES]",
      "    if PROBLEM_TYPE in ('classification','regression','forecasting'):",
      "        if TARGET not in df.columns: raise ValueError(f'Missing target: {TARGET}')",
      "        pred = model.predict(X)",
      "        print('Model:', MODEL_NAME, '| tested rows:', len(X), '| sample predictions:', pred[:10].tolist())",
      "        from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score",
      "        y = df[TARGET]",
      "        if PROBLEM_TYPE == 'classification': print(json.dumps({'accuracy':float(accuracy_score(y,pred)),'f1_weighted':float(f1_score(y,pred,average='weighted',zero_division=0))},indent=2))",
      "        else: print(json.dumps({'mae':float(mean_absolute_error(y,pred)),'rmse':float(mean_squared_error(y,pred)**0.5),'r2':float(r2_score(y,pred))},indent=2))",
      "    elif hasattr(model,'predict'): print('Output sample:', model.predict(X)[:10].tolist())",
      "    elif hasattr(model,'transform'): print('Transformed output:', model.transform(X)[:5].tolist())",
      "    else:",
      "        est=model.named_steps['model']; prep=model.named_steps['preprocess']; Xt=prep.transform(X)",
      "        if hasattr(est,'fit_predict'): out=est.fit_predict(Xt)",
      "        elif hasattr(est,'decision_function'): out=est.decision_function(Xt)",
      "        else: raise RuntimeError('This estimator cannot output predictions for new rows.')",
      "        print('Unsupervised output sample:', out[:10].tolist())",
      "if __name__ == '__main__': main()",
    ]
    Path(path).write_text("\n".join(lines)+"\n",encoding="utf-8")
