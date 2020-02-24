#1) Download the data in the Data/Financial Data
#folder on Canvas. Open the file and take a peek at
#the data dictionary. What do you think this data is used for?

#This data seems like it could be used to determine
#if someone should be given a loan

import pandas as pd
import matplotlib.pyplot as plt

#2) Create a function called loadAndCleanData
#that takes as an argument a filename and returns
#a Pandas dataframe. That dataframe should contain
#the data from the CSV file cleaned such that any
#cells missing data, containing a NaN value or the
#string "NA" are filled with 0s (this is a technique
#called zero-filling that we will talk about shortly!)
def loadAndCleanData(fileName):
    openFile = pd.read_csv(fileName)
    openFile.fillna(value = 0, inplace = True)
    return openFile

#3) Add a line to your Python file that uses the function
#to load in the creditData.csv file from Canvas when the
#Python script is run.
newData = loadAndCleanData("creditData.csv")

#4) Now that you've got your data loading, you can generate
#probability density functions for each feature. These PDFs
#will tell you the probability of a given feature occurring
#based on our data. You can use Kernel Density Estimation (KDE)
#to do this. Write a function called computePDF that takes as
#arguments a target feature and a dataset and generates a KDE plot
#for each feature in your data (hint: check out the plot.kde function
#here (Links to an external site.)). You will need to import
#matplotlib.pyplot as plt and use plt.show() to make the graphs appear.
#Call that feature on each column of your dataset when you run your script.
def computePDF(columnName, dataSet):
    newPlot = dataSet[columnName].plot.kde()
    plt.show(newPlot)


headers = list(newData.columns.values)
for i in headers:
    computePDF(i, newData)


#5) Given the skews that you see in your data, you might want to step
#back and take a look at what's actually in your data. You can look
#at the distribution of values in the columns. This will help you
#understand what data you have. To do this, write a function called
#viewDistribution that takes in the name of a column and a dataframe
#and shows a histogram of values in that column (hint: check out the
#hist function here (Links to an external site.)).  Comment out your
#computePDF function call and instead use viewDistribution to look at
#the distribution of each column in your dataset when the Python script
#is run. This should come after you call the loadAndCleanData function.
#Notice anything strange about some of these histograms?
def viewDistribution(columnName, dataSet):
    newPlot = dataSet.hist(column = columnName)
    plt.show(newPlot)


columns = list(newData.columns.values)
for i in columns:
    viewDistribution(i, newData)


#6)When your data distributions are radically skewed, you can use a log
#scale to help reveal data that is otherwise too sparse to see.
#Write a new version of the viewDistribution function called
#viewLogDistribution to show the log distribution of each column.
#Add this function call after your viewDistribution call to view the
#regular and log distributions of each feature.
def viewLogDistribution(columnName, dataSet):
    newPlot = dataSet.hist(column = columnName, log = True)
    plt.show(newPlot)

logColumns = list(newData.columns.values)

for i in logColumns:
    viewLogDistribution(i, newData)


#7) Use the two distributions to identify three bins per column that
#divide your data into roughly equal numbers. What are those bins? Note
#you do not need bins for "SeriousDlqin2yrs" as that is the feature you
#are modeling (it is your dependent variable)

def bins(columnName, dataSet):
    newBins = pd.qcut(dataSet[columnName], q = 3, duplicates = 'drop')
    print(newBins[0])
    return newBins

columnNames = list(newData.columns.values)
for i in columnNames:
    newBins = bins(i, newData)

#8) Write a function called computeDefaultRisk that takes four arguments---
#a column name, a bin (as an array [start,end]), a target feature, and a
#dataframe---and returns the probability that someone will be at least 90
#days delinquent on their account (in other words, "SeriousDlqin2yrs" = 1).
#Keep in mind that this probability is conditional, that means you'll want to
#use the equation for conditional probabilities to compute it. In plain English,
#you should compute the probability that a loan will become seriously delinquent
#given your target feature falls into the bin range. For example, if I'm looking
#at ages between 0 and 40, I want to compute the probability that a loan will go
#into serious delinquency given the applicant is between 0 and 40.

#P(A | B)
# P((B and A) / B)

def computeDefaultRisk(columnName, bin, target, dataSet):
    ageBetween = 0
    isDelqCount = 0
    for i,row in dataSet.iterrows():
        if row[target] >= bin[0] and row[target] < bin[1]:
            ageBetween += 1
            if row[columnName] == 1:
                isDelqCount +=1
    allData = len(newData)
    probA = ageBetween/allData
    probB = isDelqCount/allData
    print(probB/probA)
    return probB/probA

#9) 9. Print out the risk of default for each of the feature bins in your
#dataset. Note it's helpful to label these with the feature and bins such
#that you can better understand your output.

defaultRisk = computeDefaultRisk("SeriousDlqin2yrs", [0,40], "age", newData)


#10)In your main file, use your loadAndCleanData function to load in
#newLoans.csv.

loans = loadAndCleanData("newLoans.csv")

#11) Use your conditional probabilities to predict the probability of default
#for each row in your CSV file. To do this, write a function called
#predictDefaultRisk that takes a row from your dataset as a parameter and
#returns the risk of default based on that data and the probabilities you
#computed from creditData.csv (hint: you might want to have predictDefaultRisk
#take a second parameter representing the risk of default for various data
#features computed from creditData.csv). You will want to compute the risk of
#default using a weighted sum with the following weights:

def predictDefaultRisk(row, bin):
    if row["age"] >= bin[0] and row["age"] < bin[1]:
        riskofDefault = defaultRisk*((row["age"]*.025)+(row["NumberOfDependents"]*.025)+(row["MonthlyIncome"]*.1)+(row["DebtRatio"]*.1)+(row["RevolvingUtilizationOfUnsecuredLines"]*.1)+(row["NumberOfOpenCreditLinesAndLoans"]*.1)+(row["NumberRealEstateLoansOrLines"]*.1)+(row["NumberOfTime30-59DaysPastDueNotWorse"]*.15)+(row["NumberOfTime60-89DaysPastDueNotWorse"]*.15)+(row["NumberOfTimes90DaysLate"]*.15))
    else:
        return 0
    print(riskofDefault)
    return riskofDefault

#12) Store the result of this function in the SeriousDlqin2yrs column.

for i,row in newData.iterrows():
    SeriousDlqin2yrs = predictDefaultRisk(row, [0,40])

#13)Plot the distribution of risks using your computePDF function.
#What do you notice about this distribution?

for i in headers:
    computePDF(i, loans)
