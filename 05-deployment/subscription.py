import pickle

from flask import Flask, request, jsonify

with open("dv.bin", 'rb') as picklefile:
    dv = pickle.load(picklefile)
with open("model2.bin", 'rb') as picklefile:
    model = pickle.load(picklefile)

def predict_single(client, dv, model):
    X = dv.transform([client])
    y_pred = model.predict_proba(X)
    return y_pred[0][1]

app = Flask('credit')

@app.route('/predict', methods=['POST'])
def predict():
    client = request.get_json()
    prediction = predict_single(client, dv, model)
    credit = prediction >= 0.5
    result = {
        'credit_probability': float(prediction),
        'creadit': bool(credit)
    }
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)