"""
VigilNet — Network Intrusion Detection System (v2.0)
Multi-Model ML Engine with Comparison & Bulk Testing
Author  : Yashank Sharma
College : Jaipur National University
Batch   : 2022–26
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, f1_score, precision_score, recall_score
)
import joblib, json, warnings
warnings.filterwarnings('ignore')

ATTACK_CATEGORIES = {
    'normal':'Normal','neptune':'DoS','back':'DoS','land':'DoS','pod':'DoS',
    'smurf':'DoS','teardrop':'DoS','mailbomb':'DoS','apache2':'DoS',
    'processtable':'DoS','udpstorm':'DoS','ipsweep':'Probe','nmap':'Probe',
    'portsweep':'Probe','satan':'Probe','mscan':'Probe','saint':'Probe',
    'ftp_write':'R2L','guess_passwd':'R2L','imap':'R2L','multihop':'R2L',
    'phf':'R2L','spy':'R2L','warezclient':'R2L','warezmaster':'R2L',
    'sendmail':'R2L','named':'R2L','snmpgetattack':'R2L','snmpguess':'R2L',
    'xlock':'R2L','xsnoop':'R2L','httptunnel':'R2L','buffer_overflow':'U2R',
    'loadmodule':'U2R','perl':'U2R','rootkit':'U2R','ps':'U2R',
    'sqlattack':'U2R','xterm':'U2R'
}
CAT_COLS  = ['protocol_type','service','flag']
PROTOCOLS = ['tcp','udp','icmp']
SERVICES  = ['http','ftp','smtp','ssh','dns','telnet','other']
FLAGS     = ['SF','S0','REJ','RSTO','RSTOS0','SH']

ATTACK_SCENARIOS = {
    'syn_flood_dos':{
        'name':'SYN Flood (DoS)',
        'description':'High-volume SYN packets with no response — classic DoS pattern',
        'data':{'duration':0,'protocol_type':'tcp','service':'http','flag':'S0',
            'src_bytes':0,'dst_bytes':0,'land':0,'wrong_fragment':0,'urgent':0,'hot':0,
            'num_failed_logins':0,'logged_in':0,'num_compromised':0,'root_shell':0,
            'su_attempted':0,'num_root':0,'num_file_creations':0,'num_shells':0,
            'num_access_files':0,'num_outbound_cmds':0,'is_host_login':0,'is_guest_login':0,
            'count':511,'srv_count':511,'serror_rate':1.0,'srv_serror_rate':1.0,
            'rerror_rate':0.0,'srv_rerror_rate':0.0,'same_srv_rate':1.0,'diff_srv_rate':0.0,
            'srv_diff_host_rate':0.0,'dst_host_count':255,'dst_host_srv_count':255,
            'dst_host_same_srv_rate':1.0,'dst_host_diff_srv_rate':0.0,
            'dst_host_same_src_port_rate':1.0,'dst_host_srv_diff_host_rate':0.0,
            'dst_host_serror_rate':1.0,'dst_host_srv_serror_rate':1.0,
            'dst_host_rerror_rate':0.0,'dst_host_srv_rerror_rate':0.0}
    },
    'port_scan_probe':{
        'name':'Port Scan (Probe)',
        'description':'Systematic scanning of ports to find vulnerabilities',
        'data':{'duration':0,'protocol_type':'tcp','service':'other','flag':'REJ',
            'src_bytes':0,'dst_bytes':0,'land':0,'wrong_fragment':0,'urgent':0,'hot':0,
            'num_failed_logins':0,'logged_in':0,'num_compromised':0,'root_shell':0,
            'su_attempted':0,'num_root':0,'num_file_creations':0,'num_shells':0,
            'num_access_files':0,'num_outbound_cmds':0,'is_host_login':0,'is_guest_login':0,
            'count':200,'srv_count':10,'serror_rate':0.0,'srv_serror_rate':0.0,
            'rerror_rate':1.0,'srv_rerror_rate':1.0,'same_srv_rate':0.05,'diff_srv_rate':0.95,
            'srv_diff_host_rate':0.98,'dst_host_count':255,'dst_host_srv_count':10,
            'dst_host_same_srv_rate':0.04,'dst_host_diff_srv_rate':0.96,
            'dst_host_same_src_port_rate':0.01,'dst_host_srv_diff_host_rate':0.98,
            'dst_host_serror_rate':0.0,'dst_host_srv_serror_rate':0.0,
            'dst_host_rerror_rate':0.99,'dst_host_srv_rerror_rate':0.99}
    },
    'brute_force_r2l':{
        'name':'Brute Force Login (R2L)',
        'description':'Repeated failed login attempts from remote machine',
        'data':{'duration':5,'protocol_type':'tcp','service':'ftp','flag':'SF',
            'src_bytes':328,'dst_bytes':0,'land':0,'wrong_fragment':0,'urgent':0,'hot':0,
            'num_failed_logins':5,'logged_in':0,'num_compromised':0,'root_shell':0,
            'su_attempted':0,'num_root':0,'num_file_creations':0,'num_shells':0,
            'num_access_files':0,'num_outbound_cmds':0,'is_host_login':0,'is_guest_login':0,
            'count':1,'srv_count':1,'serror_rate':0.0,'srv_serror_rate':0.0,
            'rerror_rate':0.0,'srv_rerror_rate':0.0,'same_srv_rate':1.0,'diff_srv_rate':0.0,
            'srv_diff_host_rate':0.0,'dst_host_count':1,'dst_host_srv_count':1,
            'dst_host_same_srv_rate':1.0,'dst_host_diff_srv_rate':0.0,
            'dst_host_same_src_port_rate':1.0,'dst_host_srv_diff_host_rate':0.0,
            'dst_host_serror_rate':0.0,'dst_host_srv_serror_rate':0.0,
            'dst_host_rerror_rate':0.0,'dst_host_srv_rerror_rate':0.0}
    },
    'buffer_overflow_u2r':{
        'name':'Buffer Overflow (U2R)',
        'description':'Attempt to gain root shell via buffer overflow exploit',
        'data':{'duration':0,'protocol_type':'tcp','service':'telnet','flag':'SF',
            'src_bytes':1408,'dst_bytes':3664,'land':0,'wrong_fragment':0,'urgent':0,'hot':2,
            'num_failed_logins':0,'logged_in':1,'num_compromised':1,'root_shell':1,
            'su_attempted':0,'num_root':1,'num_file_creations':1,'num_shells':2,
            'num_access_files':0,'num_outbound_cmds':0,'is_host_login':0,'is_guest_login':0,
            'count':1,'srv_count':1,'serror_rate':0.0,'srv_serror_rate':0.0,
            'rerror_rate':0.0,'srv_rerror_rate':0.0,'same_srv_rate':1.0,'diff_srv_rate':0.0,
            'srv_diff_host_rate':0.0,'dst_host_count':1,'dst_host_srv_count':1,
            'dst_host_same_srv_rate':1.0,'dst_host_diff_srv_rate':0.0,
            'dst_host_same_src_port_rate':1.0,'dst_host_srv_diff_host_rate':0.0,
            'dst_host_serror_rate':0.0,'dst_host_srv_serror_rate':0.0,
            'dst_host_rerror_rate':0.0,'dst_host_srv_rerror_rate':0.0}
    },
    'normal_http':{
        'name':'Normal HTTP Browse',
        'description':'Standard legitimate web browsing connection',
        'data':{'duration':8,'protocol_type':'tcp','service':'http','flag':'SF',
            'src_bytes':22543,'dst_bytes':9898,'land':0,'wrong_fragment':0,'urgent':0,'hot':0,
            'num_failed_logins':0,'logged_in':1,'num_compromised':0,'root_shell':0,
            'su_attempted':0,'num_root':0,'num_file_creations':0,'num_shells':0,
            'num_access_files':0,'num_outbound_cmds':0,'is_host_login':0,'is_guest_login':0,
            'count':6,'srv_count':6,'serror_rate':0.0,'srv_serror_rate':0.0,
            'rerror_rate':0.0,'srv_rerror_rate':0.0,'same_srv_rate':1.0,'diff_srv_rate':0.0,
            'srv_diff_host_rate':0.0,'dst_host_count':64,'dst_host_srv_count':64,
            'dst_host_same_srv_rate':1.0,'dst_host_diff_srv_rate':0.0,
            'dst_host_same_src_port_rate':0.0,'dst_host_srv_diff_host_rate':0.0,
            'dst_host_serror_rate':0.0,'dst_host_srv_serror_rate':0.0,
            'dst_host_rerror_rate':0.0,'dst_host_srv_rerror_rate':0.0}
    }
}

class DataGenerator:
    @staticmethod
    def generate(n_samples=5000, random_state=42):
        np.random.seed(random_state)
        n = n_samples
        attack_types = [a for a in ATTACK_CATEGORIES if a != 'normal']
        normal_mask  = np.random.random(n) < 0.60
        data = {
            'duration':np.random.exponential(5,n).astype(int),
            'protocol_type':np.random.choice(PROTOCOLS,n),
            'service':np.random.choice(SERVICES,n),
            'flag':np.random.choice(FLAGS,n),
            'src_bytes':np.random.exponential(5000,n).astype(int),
            'dst_bytes':np.random.exponential(3000,n).astype(int),
            'land':np.random.binomial(1,0.01,n),'wrong_fragment':np.random.poisson(0.1,n),
            'urgent':np.random.poisson(0.01,n),'hot':np.random.poisson(1,n),
            'num_failed_logins':np.random.poisson(0.1,n),
            'logged_in':np.random.binomial(1,0.7,n),
            'num_compromised':np.random.poisson(0.5,n),
            'root_shell':np.random.binomial(1,0.02,n),
            'su_attempted':np.random.binomial(1,0.01,n),
            'num_root':np.random.poisson(0.3,n),
            'num_file_creations':np.random.poisson(0.2,n),
            'num_shells':np.random.poisson(0.05,n),
            'num_access_files':np.random.poisson(0.3,n),
            'num_outbound_cmds':np.zeros(n,dtype=int),
            'is_host_login':np.random.binomial(1,0.01,n),
            'is_guest_login':np.random.binomial(1,0.02,n),
            'count':np.random.randint(1,512,n),'srv_count':np.random.randint(1,512,n),
            'serror_rate':np.random.uniform(0,1,n),'srv_serror_rate':np.random.uniform(0,1,n),
            'rerror_rate':np.random.uniform(0,1,n),'srv_rerror_rate':np.random.uniform(0,1,n),
            'same_srv_rate':np.random.uniform(0,1,n),'diff_srv_rate':np.random.uniform(0,1,n),
            'srv_diff_host_rate':np.random.uniform(0,1,n),
            'dst_host_count':np.random.randint(1,256,n),
            'dst_host_srv_count':np.random.randint(1,256,n),
            'dst_host_same_srv_rate':np.random.uniform(0,1,n),
            'dst_host_diff_srv_rate':np.random.uniform(0,1,n),
            'dst_host_same_src_port_rate':np.random.uniform(0,1,n),
            'dst_host_srv_diff_host_rate':np.random.uniform(0,1,n),
            'dst_host_serror_rate':np.random.uniform(0,1,n),
            'dst_host_srv_serror_rate':np.random.uniform(0,1,n),
            'dst_host_rerror_rate':np.random.uniform(0,1,n),
            'dst_host_srv_rerror_rate':np.random.uniform(0,1,n),
            'label':np.where(normal_mask,'normal',np.random.choice(attack_types,n))
        }
        return pd.DataFrame(data)

    @staticmethod
    def random_sample():
        np.random.seed(None)
        return {k:v for k,v in {
            'duration':int(np.random.exponential(5)),
            'protocol_type':np.random.choice(PROTOCOLS),
            'service':np.random.choice(SERVICES),
            'flag':np.random.choice(FLAGS),
            'src_bytes':int(np.random.exponential(5000)),
            'dst_bytes':int(np.random.exponential(3000)),
            'land':0,'wrong_fragment':0,'urgent':0,
            'hot':int(np.random.poisson(1)),
            'num_failed_logins':0,'logged_in':int(np.random.binomial(1,0.7)),
            'num_compromised':0,'root_shell':0,'su_attempted':0,'num_root':0,
            'num_file_creations':0,'num_shells':0,'num_access_files':0,'num_outbound_cmds':0,
            'is_host_login':0,'is_guest_login':0,
            'count':int(np.random.randint(1,512)),
            'srv_count':int(np.random.randint(1,512)),
            'serror_rate':round(float(np.random.uniform(0,1)),4),
            'srv_serror_rate':round(float(np.random.uniform(0,1)),4),
            'rerror_rate':round(float(np.random.uniform(0,1)),4),
            'srv_rerror_rate':round(float(np.random.uniform(0,1)),4),
            'same_srv_rate':round(float(np.random.uniform(0,1)),4),
            'diff_srv_rate':round(float(np.random.uniform(0,1)),4),
            'srv_diff_host_rate':round(float(np.random.uniform(0,1)),4),
            'dst_host_count':int(np.random.randint(1,256)),
            'dst_host_srv_count':int(np.random.randint(1,256)),
            'dst_host_same_srv_rate':round(float(np.random.uniform(0,1)),4),
            'dst_host_diff_srv_rate':round(float(np.random.uniform(0,1)),4),
            'dst_host_same_src_port_rate':round(float(np.random.uniform(0,1)),4),
            'dst_host_srv_diff_host_rate':round(float(np.random.uniform(0,1)),4),
            'dst_host_serror_rate':round(float(np.random.uniform(0,1)),4),
            'dst_host_srv_serror_rate':round(float(np.random.uniform(0,1)),4),
            'dst_host_rerror_rate':round(float(np.random.uniform(0,1)),4),
            'dst_host_srv_rerror_rate':round(float(np.random.uniform(0,1)),4),
        }.items()}


class NIDSEngine:
    MODEL_DEFS = {
        'random_forest':{'name':'Random Forest','cls':RandomForestClassifier,'kwargs':{'n_estimators':100,'max_depth':15,'random_state':42,'n_jobs':-1}},
        'svm':{'name':'SVM','cls':SVC,'kwargs':{'kernel':'rbf','C':1.0,'probability':True,'random_state':42}},
        'knn':{'name':'KNN','cls':KNeighborsClassifier,'kwargs':{'n_neighbors':5,'n_jobs':-1}},
        'naive_bayes':{'name':'Naive Bayes','cls':GaussianNB,'kwargs':{}},
    }

    def __init__(self):
        self.trained_models={}; self.scaler=StandardScaler()
        self.le_dict={}; self.label_encoder=LabelEncoder()
        self.is_trained=False; self.metrics={}; self.feature_cols=[]
        self.active_model='random_forest'

    def _encode_cats(self, df, fit=False):
        df=df.copy()
        for col in CAT_COLS:
            if fit:
                self.le_dict[col]=LabelEncoder()
                df[col]=self.le_dict[col].fit_transform(df[col].astype(str))
            else:
                known=set(self.le_dict[col].classes_)
                df[col]=df[col].apply(lambda x: x if x in known else self.le_dict[col].classes_[0])
                df[col]=self.le_dict[col].transform(df[col].astype(str))
        return df

    def preprocess(self, df, fit=False):
        df=self._encode_cats(df,fit=fit)
        feat_cols=[c for c in df.columns if c!='label']
        X=df[feat_cols].values.astype(float)
        if fit: self.feature_cols=feat_cols; X=self.scaler.fit_transform(X)
        else: X=self.scaler.transform(X)
        return X, feat_cols

    def train_all(self, df):
        y_raw=df['label'].apply(lambda x: ATTACK_CATEGORIES.get(x.strip(),'Unknown'))
        y=self.label_encoder.fit_transform(y_raw)
        X,_=self.preprocess(df,fit=True)
        X_tr,X_te,y_tr,y_te=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
        self.metrics={'classes':self.label_encoder.classes_.tolist(),'train_size':len(X_tr),'test_size':len(X_te),'models':{}}
        for key,info in self.MODEL_DEFS.items():
            print(f'[*] Training {info["name"]} ...')
            clf=info['cls'](**info['kwargs']); clf.fit(X_tr,y_tr)
            self.trained_models[key]=clf
            y_pred=clf.predict(X_te)
            acc=accuracy_score(y_te,y_pred)
            prec=precision_score(y_te,y_pred,average='weighted',zero_division=0)
            rec=recall_score(y_te,y_pred,average='weighted',zero_division=0)
            f1=f1_score(y_te,y_pred,average='weighted',zero_division=0)
            cm=confusion_matrix(y_te,y_pred)
            report=classification_report(y_te,y_pred,target_names=self.label_encoder.classes_,output_dict=True,zero_division=0)
            feat_imp={}
            if key=='random_forest':
                feat_imp=dict(zip(self.feature_cols,clf.feature_importances_.tolist()))
            self.metrics['models'][key]={
                'name':info['name'],'accuracy':round(acc*100,2),
                'precision':round(prec*100,2),'recall':round(rec*100,2),
                'f1_score':round(f1*100,2),'confusion_matrix':cm.tolist(),
                'classification_report':report,'feature_importance':feat_imp,
                'top_features':sorted(feat_imp.items(),key=lambda x:x[1],reverse=True)[:10] if feat_imp else []
            }
            print(f'    {info["name"]} Acc={acc*100:.2f}%')
        self.is_trained=True; return self.metrics

    def predict(self, sample, model_key=None):
        key=model_key or self.active_model; clf=self.trained_models[key]
        df=pd.DataFrame([sample]); X,_=self.preprocess(df,fit=False)
        idx=clf.predict(X)[0]; label=self.label_encoder.classes_[idx]
        proba=clf.predict_proba(X)[0]; conf=float(proba[idx])
        return {'prediction':label,'confidence':round(conf*100,2),
                'is_attack':label!='Normal',
                'probabilities':{c:round(float(p),4) for c,p in zip(self.label_encoder.classes_,proba)},
                'model_used':self.MODEL_DEFS[key]['name']}

    def predict_all_models(self, sample):
        return {key:self.predict(sample,model_key=key) for key in self.trained_models}

    def predict_bulk(self, df, model_key=None):
        key=model_key or self.active_model; clf=self.trained_models[key]
        has_label='label' in df.columns
        df_feat=df.drop(columns=['label'],errors='ignore')
        X,_=self.preprocess(df_feat,fit=False)
        preds=clf.predict(X); probas=clf.predict_proba(X)
        results=[]
        for i,(pi,proba) in enumerate(zip(preds,probas)):
            label=self.label_encoder.classes_[pi]; conf=float(proba[pi])
            row={'row':i+1,'prediction':label,'confidence':round(conf*100,2),'is_attack':label!='Normal'}
            if has_label:
                actual=ATTACK_CATEGORIES.get(str(df['label'].iloc[i]).strip(),'Unknown')
                row['actual']=actual; row['correct']=(label==actual)
            results.append(row)
        return results

    def batch_test(self, n=100):
        df=DataGenerator.generate(n_samples=n,random_state=None)
        results=self.predict_bulk(df)
        correct=sum(1 for r in results if r.get('correct',False))
        dist={}
        for r in results: dist[r['prediction']]=dist.get(r['prediction'],0)+1
        return {'total':n,'correct':correct,'accuracy':round(correct/n*100,2),'results':results[:50],'distribution':dist}

    def save(self, path='vigilnet_models.pkl'):
        joblib.dump({'trained_models':self.trained_models,'scaler':self.scaler,
                     'le_dict':self.le_dict,'label_encoder':self.label_encoder,
                     'metrics':self.metrics,'feature_cols':self.feature_cols},path)
        print(f'[+] Saved to {path}')

    def load(self, path='vigilnet_models.pkl'):
        d=joblib.load(path)
        self.trained_models=d['trained_models']; self.scaler=d['scaler']
        self.le_dict=d['le_dict']; self.label_encoder=d['label_encoder']
        self.metrics=d['metrics']; self.feature_cols=d['feature_cols']
        self.is_trained=True; print(f'[+] Loaded from {path}')


if __name__=='__main__':
    print('='*55)
    print('  VigilNet v2.0 — Multi-Model NIDS')
    print('  Author  : Yashank Sharma')
    print('  College : Jaipur National University')
    print('  Batch   : 2022-26')
    print('='*55)
    df=DataGenerator.generate(5000)
    engine=NIDSEngine(); engine.train_all(df)
    print('\n── Model Comparison ──')
    for k,m in engine.metrics['models'].items():
        print(f'  {m["name"]:<16} Acc={m["accuracy"]}%  F1={m["f1_score"]}%')
    engine.save('vigilnet_models.pkl')
    with open('metrics.json','w') as f: json.dump(engine.metrics,f,indent=2,default=str)
    print('\n[+] Done!')
