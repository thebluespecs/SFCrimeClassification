import pandas as pd
import pickle

test = pd.read_csv('test.csv')
Xtest = pd.read_csv('Xtest.csv')
filename = 'PickleFile.pkl'
pkl = open(filename, 'rb')
model = pickle.load(pkl)
print ("Loaded XGBoost model\n", model)
predicted = model.predict_proba(Xtest)
col = ['ARSON','ASSAULT', 'BAD CHECKS','BRIBERY','BURGLARY','DISORDERLY CONDUCT',
       'DRIVING UNDER THE INFLUENCE','DRUG/NARCOTIC','DRUNKENNESS','EMBEZZLEMENT',
       'EXTORTION','FAMILY OFFENSES','FORGERY/COUNTERFEITING','FRAUD','GAMBLING',
       'KIDNAPPING','LARCENY/THEFT','LIQUOR LAWS','LOITERING','MISSING PERSON','NON-CRIMINAL',
       'OTHER OFFENSES','PROSTITUTION','RECOVERED VEHICLE','ROBBERY','RUNAWAY','SECONDARY CODES',
       'SEX OFFENSES FORCIBLE','STOLEN PROPERTY','SUICIDE','SUSPICIOUS OCC','TRESPASS','VANDALISM',
       'VEHICLE THEFT','WARRANTS','WEAPON LAWS']
result = pd.DataFrame(data = predicted, columns = col )
result.insert(0,"Id" , test['Id'])
result.to_csv('Result.csv', encoding='utf-8', index=False)
