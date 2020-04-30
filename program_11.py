#!/bin/env python
# 
# ABE65100
# Lab 11
#
# Author: Ankit Ghanghas
#
# This script is designed to do assignment 11 for the Environemtal Informatics
# course. This script reads in Annual and Monthly metric of discharge dataset
# produced for two different sites in Lab 10 and makes presentation graphics
# for the two input datasets.


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats


def ReadData( fileName ):
    """This function takes a filename as input, and returns a dataframe with
    raw data read from that file in a Pandas DataFrame.  The DataFrame index
    should be the year, month and day of the observation.  DataFrame headers
    should be "agency_cd", "site_no", "Date", "Discharge", "Quality". The 
    "Date" column should be used as the DataFrame index. The pandas read_csv
    function will automatically replace missing values with np.NaN, but needs
    help identifying other flags used by the USGS to indicate no data is 
    availabiel.  Function returns the completed DataFrame, and a dictionary 
    designed to contain all missing value counts that is initialized with
    days missing between the first and last date of the file."""
    
    # define column names
    colNames = ['agency_cd', 'site_no', 'Date', 'Discharge', 'Quality']

    # open and read the file
    DataDF = pd.read_csv(fileName, header=1, names=colNames,  
                         delimiter=r"\s+",parse_dates=[2], comment='#',
                         na_values=['Eqp'])
    DataDF = DataDF.set_index('Date')

   
    # quantify the number of missing values
    MissingValues = DataDF["Discharge"].isna().sum()
  
    DataDF.loc[~(DataDF['Discharge']>0),'Discharge']=np.nan  
    #DataDF=DataDF.dropna()

    return( DataDF, MissingValues )

def ClipData( DataDF, startDate, endDate ):
    """This function clips the given time series dataframe to a given range 
    of dates. Function returns the clipped dataframe and and the number of 
    missing values."""
    

    DataDF=DataDF.loc[startDate:endDate]
    MissingValues = DataDF['Discharge'].isna().sum()
 
    return( DataDF, MissingValues )


def ReadMetrics( fileName ):
    """This function takes a filename as input, and returns a dataframe with
    the metrics from the assignment on descriptive statistics and 
    environmental metrics.  Works for both annual and monthly metrics. 
    Date column should be used as the index for the new dataframe.  Function 
    returns the completed DataFrame."""

    DataDF = pd.read_csv(fileName, header =0, delimiter=',', parse_dates=['Date'], index_col=['Date'])
    
    return( DataDF )

def GetMonthlyStatistics(DataDF):

    col_name = ['site_no', 'Mean Flow']
    monthly_data=DataDF.resample('MS').mean()
    MoDataDF=pd.DataFrame(index=monthly_data.index,columns=col_name)
    
    MoDataDF['site_no']=DataDF.resample('MS')['site_no'].mean()
    MoDataDF['Mean Flow']=DataDF.resample('MS')['Discharge'].mean()

    return (MoDataDF)

def GetMonthlyAverages(MoDataDF):
    """ 
    This Function calculates the annual average monthly values for all statistics and metrics.
    The routine returns an array of mean values for each metric in the original dataframe.
    """

    col_name = ['site_no', 'Mean Flow']
    MonthlyAverages= pd.DataFrame(0, index=range(1,13), columns = col_name)
    a=[3,4,5,6,7,8,9,10,11,0,1,2]
    idx=0
    
    for i in range(12):
        MonthlyAverages.iloc[idx,0] = MoDataDF['site_no'][::12].mean()
        MonthlyAverages.iloc[idx,1] = MoDataDF['Mean Flow'][a[idx]::12].mean()
        idx +=1

    return (MonthlyAverages)

# the following condition checks whether we are running as a script, in which 
# case run the test code, otherwise functions are being imported so do not.
# put the main routines from your code after this conditional check.

if __name__ == '__main__':

    # define full river names as a dictionary so that abbreviations are not used in figures
    riverName = { "Wildcat": "Wildcat Creek",
                  "Tippe": "Tippecanoe River" }
    
    fileName = { "Wildcat": "WildcatCreek_Discharge_03335000_19540601-20200315.txt",
                 "Tippe": "TippecanoeRiver_Discharge_03331500_19431001-20200315.txt" }
    
    # define blank dictionaries (these will use the same keys as fileName)
    DataDF = {}
    MissingValues = {}
    MoDataDF = {}
    MonthlyAverages = {}
    
    # process input datasets
    for file in fileName.keys():
        
        DataDF[file], MissingValues[file] = ReadData(fileName[file])
        
        # clip to consistent period
        DataDF[file], MissingValues[file] = ClipData( DataDF[file], '1969-10-01', '2019-09-30' )
        
        # Monthly Metrics
        MoDataDF[file] = GetMonthlyStatistics(DataDF[file])
        
        # annual average monthly values for metrics
        MonthlyAverages[file] = GetMonthlyAverages(MoDataDF[file])
 
    AnMetrics=ReadMetrics('Annual_Metrics.csv')
    MoMatrics=ReadMetrics('Monthly_Metrics.csv')
    
    AnTippe=AnMetrics[AnMetrics['Station']=='Tippe'] # get annual metrics for only Tippecanoe station
    AnWildcat=AnMetrics[AnMetrics['Station']=='Wildcat'] # get annual metrics for only Wildcat station

    Clip_Tippe=DataDF['Tippe']['2014-10-01' : '2019-09-30'] # clip tippe data for last 5 years.
    Clip_Wild=DataDF['Wildcat']['2014-10-01' : '2019-09-30'] # clip wildcat dataa for last 5 years



    #Plot daily flow for both rivers for last 5 years
    plt.figure()
    plt.plot(Clip_Tippe['Discharge'], label='Tippecanoe River', color ='blue')
    plt.plot(Clip_Wild['Discharge'], label='Wildcat Creek', color='red')
    plt.xlabel('Date')
    plt.ylabel('Discharge (cfs)')
    plt.title('Daily Flow between 2014 to 2019')
    plt.legend()
    plt.rcParams.update({'font.size':12}) # set font size as 20 for all parameters
    plt.savefig('dailyflow.png', dpi=96)
    plt.show()
    
    # Plot Annual Coefficient of Variation
    plt.figure()
    plt.plot(AnTippe['Coeff Var'], color='blue', label='Tippecanoe River', linestyle='None', marker='o')
    plt.plot(AnWildcat['Coeff Var'],  color= 'red', label= 'Wildcat Creek', linestyle='None', marker = '*')
    plt.ylabel('Coefficient of Variation')
    plt.xlabel('Date (years)')
    plt.title('Annual Coefficient of Variation')
    plt.legend(fontsize=12, loc= 'upper right')
    plt.rcParams.update({'font.size':12})
    plt.savefig('coeffvar.png', dpi= 96)
    plt.show()
    
    
    # plot TQmean
    plt.figure()
    plt.plot(AnTippe['Tqmean'], color='blue', label='Tippecanoe River')
    plt.plot(AnWildcat['Tqmean'],  color= 'red', label= 'Wildcat Creek')
    plt.ylabel('TQmean')
    plt.xlabel('Date (years)')
    plt.title('Annual TQmean')
    plt.legend(fontsize=12, loc= 'upper right')
    plt.rcParams.update({'font.size':12})
    plt.savefig('tqmean.png', dpi= 96)
    plt.show()
    
    # Plot R-B Index
    plt.figure()
    plt.plot(AnTippe['R-B Index'], color='blue', label='Tippecanoe River')
    plt.plot(AnWildcat['R-B Index'],  color= 'red', label= 'Wildcat Creek')
    plt.ylabel('R-B Index')
    plt.xlabel('Date (years)')
    plt.title('Annual R-B Index')
    plt.legend(fontsize=12)
    plt.rcParams.update({'font.size':12})
    plt.savefig('rbindex.png', dpi= 96)
    plt.show()  
    
    # Plot Average Monthly Mean Flow 
    plt.figure()
    plt.plot(MonthlyAverages['Tippe']['Mean Flow'], label='Tippecanoe River', color='blue')
    plt.plot(MonthlyAverages['Wildcat']['Mean Flow'], label='Wildcat Creek', color='red')
    plt.xlabel('Month')
    plt.ylabel('Discharge (cfs)')
    plt.title('Average Annual Monthly Flow')
    plt.legend(fontsize=12)
    plt.rcParams.update({'font.size':12})
    plt.savefig('monthly_flow.png', dpi= 96)
    plt.show()
    
    # Plot annual peak with exceedence probability.
    tip_flow=AnTippe['Peak Flow'].sort_values(ascending=False) # extracts Peak flow values form Annual Metrics and sort them in descendin order
    rank_tip=stats.rankdata(tip_flow, method='average')[::-1] # ranks the events and then reverses them so that rank 1 is highest event
    exced_prob_tip= [100*(rank_tip[i]/(len(tip_flow)+1)) for i in range(len(rank_tip))] # excedence proability of flow in percentage
    
    wild_flow=AnWildcat['Peak Flow'].sort_values(ascending=False)
    rank_wild=stats.rankdata(wild_flow, method='average')[::-1]
    exced_prob_wild= [100*(rank_wild[i]/(len(wild_flow)+1)) for i in range(len(rank_wild))]
    plt.figure()
    plt.plot(exced_prob_tip,tip_flow, label='Tippecanoe River', color='blue')
    plt.plot(exced_prob_wild, wild_flow, label='Wildcat Creek', color='red')
    plt.xlabel('Exceedance Probability (%)')
    plt.ylabel('Peak Discharge (cfs)')
    plt.title('Return Period of Annual Peak Flow Events')
    plt.xticks(range(0,100,10))
    plt.legend(fontsize=12)
    plt.rcParams.update({'font.size':12})
    plt.savefig('exceed.png', dpi= 96)
    plt.show()
