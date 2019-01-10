import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import log_loss
import pickle

class CrimeClass(object):
    def __init__(self, trainFile, testFile):
        self.trainFile = trainFile
        self.testFile = testFile
        self._xgb = XGBClassifier(
                            max_depth=7,
                            learning_rate=0.375,
                            n_estimators=110,
                            gamma=0,
                            reg_alpha =0.1,
                            objective = 'multi:softprob',
                            booster='gbtree',
                            silent=True,
                            subsample = .8,
                            colsample_bytree = 0.8,
                            max_delta_step = 1,
                            n_jobs=-1,
                            random_state = 1711
                            )
        self._lr = LogisticRegression( random_state = 1711, max_iter = 200, verbose = 1, n_jobs = -1, solver = 'sag', multi_class = 'multinomial')
        self.train_data = None
        self.train_labels = None
        self.test_data = None
        self.predicted_labels = None

    def street1_eng(self,address):
        if '/' in address:
            j=address[address.index('/')+1:]
            l=address[:address.index('/')]
            if j>l:
                return j
            else:
                return l
        else:
            k=address.split(" ")
            return k[1]

    def street2_eng(self,address):
        if '/' in address:
            j=address[address.index('/')+1:]
            l=address[:address.index('/')]
            if j>l:
                return l
            else:
                return j
        else:
            k=address.split(" ")
            return address[address.index('f')+1:]

    def minute_eng(self,minute):
        if minute>30:
            return minute-30
        else:
            return minute

    def is_weekend(self,day):
        if day == 'Friday' or day == 'Saturday' or day == 'Sunday':
            return 1
        else:
            return 0

    def StreetNum_eng(self,address):
        if '/' in address:
            return 'junction'
        else:
            k=address.split(" ")
            return k[0]

    def CountVec(self,data):
        count_vec = TfidfVectorizer(
        max_df = 0.3,
        min_df = 3,
        lowercase = True,
        ngram_range = (1,2),
        analyzer = 'word'
        )
        data_count = count_vec.fit_transform(data.Address)
        indices = pd.DataFrame(count_vec.get_feature_names())

        n_comp = 50
        svd_obj = TruncatedSVD(n_components = n_comp, algorithm = 'randomized')
        svd_obj.fit(data_count)
        data_svd = pd.DataFrame(svd_obj.transform(data_count))
        data_svd.columns = ['svd_char_' + str(i) for i in range(n_comp)]
        data = pd.concat([data, data_svd], axis=1)
        del data_count, data_svd
        return data

    def Outliers(self,data):
        data = data[(data['X']< -121)]
        data = data[(data['Y']<40)]
        return data

    def Preprocess(self,data):
        
        scaler=StandardScaler()
        scaler.fit(data[["X","Y"]])
        data[["X","Y"]]=scaler.transform(data[["X","Y"]])

        data["X_1"] = .707* data["Y"] + .707* data["X"]
        data["Y_1"] = .707* data["Y"] - .707* data["X"]
        data["X_2"] = (1.732/2)* data["X"] + (1./2)* data["Y"]
        data["Y_2"] = (1.732/2)* data["Y"] - (1./2)* data["X"]
        data["X_3"] = (1./2)* data["X"] + (1.732/2)* data["Y"]
        data["Y_3"] = (1./2)* data["Y"] - (1.732/2)* data["X"]
        data["radial_Distance"] = np.sqrt( np.power(data["Y"],2) + np.power(data["X"],2) )

        data['Dates'] = pd.to_datetime(data['Dates'])
        data['Year'] = data['Dates'].dt.year
        data['Month'] = data['Dates'].dt.month
        data['Day'] = data['Dates'].dt.day
        data['Hour'] = data['Dates'].dt.hour
        data['Minute'] = data['Dates'].dt.minute
        data=data.drop(['Dates'], axis=1)

        data['Minute'] = data['Minute'].apply(self.minute_eng)

        data['PdDistrict']=data['PdDistrict'].astype('category')
        data['PdDistrict']=data['PdDistrict'].cat.codes

        data["CrossRoad"] = data["Address"].str.contains("/")
        data["CrossRoad"]=data["CrossRoad"].astype('category')
        data["CrossRoad"]=data["CrossRoad"].cat.codes

        data["AV"] = data["Address"].str.contains("AV")
        data["AV"]=data["AV"].astype('category')
        data["AV"]=data["AV"].cat.codes

        data["DayOfWeek"]=data["DayOfWeek"].astype('category')
        data["DayOfWeek"]=data["DayOfWeek"].cat.codes

        data['StreetNo.'] = data['Address'].apply(self.StreetNum_eng)
        data["StreetNo."]=data["StreetNo."].astype('category')
        data["StreetNo."]=data["StreetNo."].cat.codes
        data = data.drop(['Address'], axis=1)
        return data

    def trainingData(self):
        df = pd.read_csv(self.trainFile)
        self.train_labels = df['Category']
        df = df.drop(['Descript', 'Resolution', 'Id', 'Category'], axis=1)
        self.train_data = self.Preprocess(df)

    def testingData(self):
        df = pd.read_csv(self.testFile)
        df = df.drop(['Descript', 'Resolution', 'Id'], axis=1)
        df = df.dropna()
        self.test_data = self.Preprocess(df)

    def data(self):
    	self.trainingData()
    	self.testingData()

    def Model_XGB(self):
        print('\nModel Training...\n')
        label_encoded_y = LabelEncoder().fit_transform(self.train_labels)
        seed = 7
        test_size = 0.15
        X_train, X_test, y_train, y_test = train_test_split(self.train_data, label_encoded_y, test_size=test_size, random_state=seed)
        self._xgb.fit(X_train, y_train, eval_set=[(X_test, y_test)], eval_metric='mlogloss',early_stopping_rounds=5, verbose=1)
        print('\nTraining Ended...\n')
        self.predicted_labels = self._xgb.predict_proba(self.test_data)
        print(("Log loss: ") + str(log_loss(y_test , self.predicted_labels)))

    def Model_LR(self):
        print('\nModel Training...\n')
        label_encoded_y = LabelEncoder().fit_transform(self.train_labels)
        seed = 7
        test_size = 0.15
        X_train, X_test, y_train, y_test = train_test_split(self.train_data, label_encoded_y, test_size=test_size, random_state=seed)
        model2.fit(X_train, y_train)
        self.predicted_labels = self._lr.predict_proba(self.test_data)
        print(("Log loss: ") + str(log_loss(y_test, self.predicted_labels)))

    def MakeSubmission(self):
        col = ['ARSON','ASSAULT', 'BAD CHECKS','BRIBERY','BURGLARY','DISORDERLY CONDUCT',
               'DRIVING UNDER THE INFLUENCE','DRUG/NARCOTIC','DRUNKENNESS','EMBEZZLEMENT',
               'EXTORTION','FAMILY OFFENSES','FORGERY/COUNTERFEITING','FRAUD','GAMBLING',
               'KIDNAPPING','LARCENY/THEFT','LIQUOR LAWS','LOITERING','MISSING PERSON','NON-CRIMINAL',
               'OTHER OFFENSES','PROSTITUTION','RECOVERED VEHICLE','ROBBERY','RUNAWAY','SECONDARY CODES',
               'SEX OFFENSES FORCIBLE','STOLEN PROPERTY','SUICIDE','SUSPICIOUS OCC','TRESPASS','VANDALISM',
               'VEHICLE THEFT','WARRANTS','WEAPON LAWS']
        result = pd.DataFrame(data = self.predicted_labels, columns = col )
        result.insert(0,"Id" , test['Id'])
        result.to_csv('Result.csv', encoding='utf-8', index=False)

    def MakePickle(self):
        filename = 'PickleFile.pkl'
        pkl = open(filename, 'wb')
        pickle.dump(self._xgb, pkl)
        pkl.close()

if __name__ == "__main__":
    train_data_name = 'train.csv'
    test_data_name = 'test.csv'
    model = CrimeClass(train_data_name,test_data_name)
    model.data()
    model.Model_XGB()
    model.MakePickle()
