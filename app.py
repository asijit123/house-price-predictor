from flask import Flask,render_template,request
import pickle
import numpy as np

app = Flask(__name__)

scale= pickle.load(open('model/scale.pkl','rb'))
ohe = pickle.load(open('model/ohe.pkl','rb'))
locality_df = pickle.load(open('model/locality_df.pkl','rb'))

reg = pickle.load(open('model/model.pkl','rb'))

status_encoder = pickle.load(open('model/status_encoder.pkl','rb'))
transaction_encoder = pickle.load(open('model/transaction_encoder.pkl','rb'))
type_encoder = pickle.load(open('model/type_encoder.pkl','rb'))
@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict',methods=['post'])
def predict():
    #recieve form data here
    area= request.form.get('area')
    bhk = int(request.form.get('bhk'))
    bathrooms = float(request.form.get('bathrooms'))
    status = request.form.get('status')
    transaction = request.form.get('transaction')
    property = request.form.get('property')
    locality = request.form.get('locality')

    #Onehot encode bhk an bathroom

    X_trans = ohe.transform(np.array([[bhk,bathrooms]])).toarray()

    #label enode status,transaction and property
    status = status_encoder.transform([status])
    status = status[0]
    transaction = transaction_encoder.transform([transaction])
    transaction = transaction[0]
    property = type_encoder.transform([property])
    property= property[0]
    #derive per_sqft and locality
    per_sqft = locality_df[locality_df['Locality'] == locality]['Per_Sqft'].mean()

    X = np.array([[area,status,transaction,property,per_sqft]])


    X = np.hstack((X,X_trans))

    X = scale.transform(X)
    y_pred = reg.predict(X)
    print(y_pred[0])

    return render_template('index.html',price=y_pred[0])
if __name__ =="__main__":
    app.run(debug=True)
