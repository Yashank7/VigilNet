"""
VigilNet v2.0 — Flask REST API
Author  : Yashank Sharma
College : Jaipur National University
Batch   : 2022–26
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import json, os, traceback
import pandas as pd

app = Flask(__name__)
CORS(app)
_engine = None

def get_engine():
    global _engine
    if _engine is None:
        from nids_model import NIDSEngine, DataGenerator
        _engine = NIDSEngine()
        if os.path.exists('vigilnet_models.pkl'):
            _engine.load('vigilnet_models.pkl')
        else:
            print('[*] Training fresh models ...')
            df = DataGenerator.generate(5000)
            _engine.train_all(df)
            _engine.save('vigilnet_models.pkl')
    return _engine

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status':'ok','version':'2.0','message':'VigilNet API running'})

@app.route('/api/train', methods=['POST'])
def train():
    try:
        from nids_model import NIDSEngine, DataGenerator
        global _engine; _engine = NIDSEngine()
        body = request.get_json(silent=True) or {}
        df   = DataGenerator.generate(int(body.get('n_samples',5000)))
        metrics = _engine.train_all(df); _engine.save('vigilnet_models.pkl')
        return jsonify({'success':True,'classes':metrics['classes'],
            'train_size':metrics['train_size'],'test_size':metrics['test_size'],
            'models':{k:{'name':v['name'],'accuracy':v['accuracy'],'precision':v['precision'],
                         'recall':v['recall'],'f1_score':v['f1_score']}
                      for k,v in metrics['models'].items()}})
    except Exception as e: return jsonify({'success':False,'error':str(e)}),500

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        engine=get_engine(); body=request.get_json()
        model_key=body.pop('model_key',None)
        result=engine.predict(body,model_key=model_key)
        return jsonify({'success':True,**result})
    except Exception as e: traceback.print_exc(); return jsonify({'success':False,'error':str(e)}),500

@app.route('/api/predict/compare', methods=['POST'])
def predict_compare():
    try:
        engine=get_engine(); sample=request.get_json()
        return jsonify({'success':True,'results':engine.predict_all_models(sample)})
    except Exception as e: return jsonify({'success':False,'error':str(e)}),500

@app.route('/api/test/batch', methods=['POST'])
def batch_test():
    try:
        engine=get_engine(); body=request.get_json(silent=True) or {}
        return jsonify({'success':True,**engine.batch_test(n=int(body.get('n',100)))})
    except Exception as e: return jsonify({'success':False,'error':str(e)}),500

@app.route('/api/test/scenarios', methods=['GET'])
def list_scenarios():
    from nids_model import ATTACK_SCENARIOS
    return jsonify({'scenarios':{k:{'name':v['name'],'description':v['description']}
                                 for k,v in ATTACK_SCENARIOS.items()}})

@app.route('/api/test/scenario/<sid>', methods=['GET'])
def test_scenario(sid):
    try:
        from nids_model import ATTACK_SCENARIOS
        if sid not in ATTACK_SCENARIOS: return jsonify({'error':'Unknown scenario'}),404
        engine=get_engine(); sc=ATTACK_SCENARIOS[sid]
        return jsonify({'success':True,'scenario':sid,'name':sc['name'],
            'description':sc['description'],'results':engine.predict_all_models(sc['data'])})
    except Exception as e: return jsonify({'success':False,'error':str(e)}),500

@app.route('/api/predict/csv', methods=['POST'])
def predict_csv():
    try:
        engine=get_engine()
        if 'file' not in request.files: return jsonify({'error':'No file uploaded'}),400
        df=pd.read_csv(request.files['file'])
        model_key=request.form.get('model_key',None)
        results=engine.predict_bulk(df,model_key=model_key)
        attacks=sum(1 for r in results if r['is_attack']); dist={}
        for r in results: dist[r['prediction']]=dist.get(r['prediction'],0)+1
        return jsonify({'success':True,'total':len(results),'attacks':attacks,
                        'normals':len(results)-attacks,'distribution':dist,'results':results[:200]})
    except Exception as e: traceback.print_exc(); return jsonify({'success':False,'error':str(e)}),500

@app.route('/api/metrics', methods=['GET'])
def metrics():
    try:
        engine=get_engine(); m=engine.metrics
        return jsonify({'success':True,'classes':m.get('classes'),
            'train_size':m.get('train_size'),'test_size':m.get('test_size'),
            'models':{k:{'name':v['name'],'accuracy':v['accuracy'],'precision':v['precision'],
                'recall':v['recall'],'f1_score':v['f1_score'],
                'confusion_matrix':v['confusion_matrix'],'top_features':v['top_features']}
                for k,v in m.get('models',{}).items()}})
    except Exception as e: return jsonify({'error':str(e)}),500

@app.route('/api/simulate', methods=['GET'])
def simulate():
    try:
        from nids_model import DataGenerator
        engine=get_engine(); sample=DataGenerator.random_sample()
        return jsonify({'success':True,'sample':sample,**engine.predict(sample)})
    except Exception as e: return jsonify({'success':False,'error':str(e)}),500

if __name__=='__main__':
    print('[*] Starting VigilNet v2.0 Flask API on http://localhost:5000')
    app.run(debug=True,port=5000)
