class visualisationBfP(object):

    def __init__(self, trainData):
		self.trainData = trainData

    def Week(self):
        crime_week = self.trainData['DayOfWeek'].value_counts()
        pylab.rcParams['figure.figsize'] = (10.0, 6.0)
        y=crime_week.keys()
        x=[]
        for i in crime_week:
            x.append(i)
        plt.bar(y,x)
        plt.xticks(rotation=30)
        plt.ylabel('no. of occurances', fontsize = 14)
        plt.title('San Franciso Crimes', fontsize = 28)

    def pdDistrict(self):
        crime_dis = self.trainData['PdDistrict'].value_counts()
        crime_dis
        pylab.rcParams['figure.figsize'] = (14.0, 6.0)
        y=crime_dis.keys()
        x=[]
        for i in crime_dis:
            x.append(i)
        plt.bar(y,x)
        plt.ylabel('no. of occurances', fontsize = 14)
        plt.title('San Franciso Crimes', fontsize = 28)

    def category(self):
        crime_sum = self.trainData['Category'].value_counts()
        pylab.rcParams['figure.figsize'] = (10.0, 12.0)
        y= np.arange(len(crime_sum.keys()))

        plt.barh(y, crime_sum.get_values(),  align='center', alpha=0.4)

        plt.yticks(y, map(lambda x:x.title(),crime_sum.keys()), fontsize = 14)
        plt.xlabel('Number of occurences', fontsize = 14)
        plt.title('San Franciso Crimes', fontsize = 28)
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0

    def XY(self):
        pylab.rcParams['figure.figsize'] = (10.0, 6.0)
        plt.scatter(self.trainData['X'], self.trainData['Y'])


class visualisationAfP(object):

    def __init__(self, trainData):
		self.trainData = trainData

    def month(self):
        crime_month = self.trainData['Month'].value_counts()
        pylab.rcParams['figure.figsize'] = (15.0, 6.0)
        months ={1:'January', 2:'February', 3:'March', 4:'April', 5:'May', 6:'June', 7:'July', 8:'August', 9:'September', 10:'October', 11:'November', 12:'December'}
        for i in range(1,13):
            crime_month[months[i]]=crime_month[i]
            del crime_month[i]

        y=crime_month.keys()
        x=[]
        for i in crime_month:
            x.append(i)
        plt.bar(y,x)
        plt.ylabel('no. of occurances', fontsize = 14)
        plt.title('San Franciso Crimes', fontsize = 28)

    def year(self):
        crime_year = self.trainData['Year'].value_counts()
        y=crime_year.keys()
        x=[]
        for i in crime_year:
            x.append(i)
        plt.bar(y,x)
        plt.xlabel('Years', fontsize = 14)
        plt.ylabel('no. of occurances', fontsize = 14)
        plt.title('San Franciso Crimes', fontsize = 28)

    def hour(self):
        crime_hour = self.trainData['Hours'].value_counts()
        y=crime_hour.keys()
        x=[]
        for i in crime_hour:
            x.append(i)
        plt.bar(y,x)
        plt.xlabel('Hours', fontsize = 14)
        plt.ylabel('no. of occurances', fontsize = 14)
        plt.title('San Franciso Crimes', fontsize = 28)

    def day(self):
        crime_day = self.trainData['Day'].value_counts()
        y=crime_day.keys()
        x=[]
        for i in crime_day:
            x.append(i)
        plt.bar(y,x)
        plt.xlabel('Days', fontsize = 14)
        plt.ylabel('no. of occurances', fontsize = 14)
        plt.title('San Franciso Crimes', fontsize = 28)

    def min(self):
        crime_min = self.trainData['Min'].value_counts()
        y=crime_min.keys()
        x=[]
        for i in crime_min:
            x.append(i)
        plt.bar(y,x)
        plt.xlabel('Minutes', fontsize = 14)
        plt.ylabel('no. of occurances', fontsize = 14)
        plt.title('San Franciso Crimes', fontsize = 28)
