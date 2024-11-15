from flask import Flask, render_template,request,jsonify
import pandas as pd
import pickle

app=Flask(__name__)
df=pd.read_csv('final_dataset.csv')
pipe=pickle.load(open("RidgeModel.pkl",'rb'))

@app.route('/')
def index():
    bedrooms = sorted(df['beds'].unique())
    bathrooms = sorted(df['baths'].unique())
    sizes = sorted(df['size'].unique())
    zip_codes =sorted(df['zip_code'].unique())

    return render_template('index.html',bedrooms=bedrooms,bathrooms=bathrooms,sizes=sizes,zip_codes=zip_codes)

@app.route('/predict',methods=['POST'])

def predict():
    bedrooms = request.form.get('beds')
    bathrooms=request.form.get('baths')
    size=request.form.get('size')
    zipcode=request.form.get('zipcode')

    # create a Dataframe with the input data
    input_data=pd.DataFrame([[bedrooms,bathrooms,size,zipcode]],columns=['beds','baths','size','zip_code'])
    print("Input Data : ")
    print(input_data)

    prediction=pipe.predict(input_data)[0]

    return str(prediction)

if __name__=="__main__":
    app.run(debug=True,port=8000) 