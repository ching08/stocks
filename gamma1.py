import pandas as pd
import numpy as np
import scipy
from scipy.stats import norm
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, date
import ipdb
import sys
import os,shutil
import os
import img2pdf
import argparse
import termcolor




# Inputs and Parameters
inputFile = 'spx_quotedata.csv'
outDir='outputs'
if os.path.exists(outDir):
    shutil.rmtree(outDir)
os.makedirs(outDir,mode=0o777)


# Black-Scholes European-Options Gamma
def calcGammaEx(S, K, vol, T, r, q, optType, OI):
    if T == 0 or vol == 0:
        return 0

    dp = (np.log(S/K) + (r - q + 0.5*vol**2)*T) / (vol*np.sqrt(T))
    dm = dp - vol*np.sqrt(T) 

    if optType == 'call':
        gamma = np.exp(-q*T) * norm.pdf(dp) / (S * vol * np.sqrt(T))
        return OI * 100 * S * S * 0.01 * gamma 
    else: # Gamma is same for calls and puts. This is just to cross-check
        gamma = K * np.exp(-r*T) * norm.pdf(dm) / (S * S * vol * np.sqrt(T))
        return OI * 100 * S * S * 0.01 * gamma 

def isThirdFriday(d):
    return d.weekday() == 4 and 15 <= d.day <= 21


def conver_csv(filename):
    # This assumes the CBOE file format hasn't been edited, i.e. table beginds at line 4
    pd.options.display.float_format = '{:,.4f}'.format
    optionsFile = open(filename)
    optionsFileData = optionsFile.readlines()
    optionsFile.close()

    # Get SPX Spot
    spotLine = optionsFileData[1]
    spotPrice = float(spotLine.split('Last:')[1].split(',')[0])
    fromStrike = 0.94 * spotPrice
    toStrike = 1.05 * spotPrice

    # Get Today's Date
    dateLine = optionsFileData[2]
    todayDate = dateLine.split('Date: ')[1].split(',')
    monthDay = todayDate[0].split(' ')

    # Handling of US/EU date formats
    if len(monthDay) == 2:
        year = int(todayDate[1].split(' ')[1])
        month = monthDay[0]
        day = int(monthDay[1])
    else:
        year=int(float(monthDay[2]))
        month=todayDate[1]
        day=int(monthDay[0])

    todayDate = datetime.strptime(month,'%B')
    todayDate = todayDate.replace(day=day, year=year)


    # Get SPX Options Data
    df = pd.read_csv(filename, sep=",", header=None, skiprows=4)
    df.columns = ['ExpirationDate','Calls','CallLastSale','CallNet','CallBid','CallAsk','CallVol',
                'CallIV','CallDelta','CallGamma','CallOpenInt','StrikePrice','Puts','PutLastSale',
                'PutNet','PutBid','PutAsk','PutVol','PutIV','PutDelta','PutGamma','PutOpenInt']

    df['ExpirationDate'] = pd.to_datetime(df['ExpirationDate'], format='%a %b %d %Y')
    df['ExpirationDate'] = df['ExpirationDate'] + timedelta(hours=16)
    df['StrikePrice'] = df['StrikePrice'].astype(float)
    df['CallIV'] = df['CallIV'].astype(float)
    df['PutIV'] = df['PutIV'].astype(float)
    df['CallGamma'] = df['CallGamma'].astype(float)
    df['PutGamma'] = df['PutGamma'].astype(float)
    df['CallOpenInt'] = df['CallOpenInt'].astype(float)
    df['PutOpenInt'] = df['PutOpenInt'].astype(float)

    # ---=== CALCULATE SPOT GAMMA ===---
    # Gamma Exposure = Unit Gamma * Open Interest * Contract Size * Spot Price 
    # To further convert into 'per 1% move' quantity, multiply by 1% of spotPrice
    df['CallGEX'] = df['CallGamma'] * df['CallOpenInt'] * 100 * spotPrice * spotPrice * 0.01
    df['PutGEX'] = df['PutGamma'] * df['PutOpenInt'] * 100 * spotPrice * spotPrice * 0.01 * -1

    df['TotalGamma'] = (df.CallGEX + df.PutGEX) / 10**9
    dfAgg = df.groupby(['StrikePrice']).sum()
    strikes = dfAgg.index.values

    return todayDate,strikes, fromStrike,toStrike, dfAgg, spotPrice, df


def error_out(msg):
    termcolor.cprint(f"ERROR: {msg}",color='red')
    sys.exit(1)
    

def extract_csv_by_date(inFile,startDatetime,endDatetime):
    outFile=os.path.join(outDir,'_tmpfile.csv')
    f1=open(outFile,'w')
    #t1=datetime.strptime(date, '%b %d %Y')
    cnt=0
    allDates=[]
    with open(inFile) as f:
        for i,line in enumerate(f.readlines()):
            if i <= 3:
                f1.write(line)
            else:
                csvDate =' '.join(line.split(',')[0].split()[1:])
                allDates.append(csvDate)
                csvDatetime = datetime.strptime(csvDate,'%b %d %Y')  
                if csvDatetime < startDatetime or csvDatetime > endDatetime:
                    continue
                else:
                    cnt += 1
                f1.write(line)
        f1.close()

    allDates1=list(dict.fromkeys(allDates))
    d1=datetime.strftime(startDatetime,'%b %d %Y')
    d2=datetime.strftime(endDatetime,'%b %d %Y')
    if cnt == 0:
        error_out(f"Did not find any data in {inFile} matches '{d1}' ~ '{d2}'\nAvialable dates are {allDates1}")
    else:
        termcolor.cprint(f"found {cnt} data in {inFile} matches '{d1}' ~ '{d2}'",color='green')

    return outFile,allDates1


def print_header(msg):
    print('*'* 120)
    print(msg)
    print('*'* 120)
    

###############
# main
###############
# validate
if not os.path.exists(inputFile):
    error_out(f"Please provide file {inputFile} in {os.path.curdir}")


tday=datetime.strftime(datetime.today(),'%b %d %Y')
parser = argparse.ArgumentParser()
parser.add_argument('-s','--start-date',default=tday,help=f"starting date, default is  %(default)s. example: python3 %(prog)s -s 'Feb 03 2023'")
parser.add_argument('-d','--duration',default=1,help="duration , default is %(default)s days. example python3 %(prog)s -s 'Feb 03 2023' -d 2")

args = parser.parse_args()

# get startdate and end date
startDate=args.start_date
startDatetime=datetime.strptime(startDate,'%b %d %Y') 
endDatetime= startDatetime + timedelta(days=int(args.duration)-1)
endDate=datetime.strftime(endDatetime,'%b %d %Y')


## extracing csv by startdate and enddate

tmpFile,allDates=extract_csv_by_date(inputFile,startDatetime,endDatetime)
todayDate,strikes,fromStrike,toStrike,dfAgg,spotPrice,df=conver_csv(tmpFile)
dateRange=f"{datetime.strftime(startDatetime,'%m/%d/%Y')}~{datetime.strftime(endDatetime,'%m/%d/%Y')}"
# Chart 1: Absolute Gamma Exposure
print_header('chart 1: Absolute Gamma Exposure')
plt.grid()
plt.bar(strikes, dfAgg['TotalGamma'].to_numpy(), width=6, linewidth=0.1, edgecolor='k', label="Gamma Exposure")
plt.xlim([fromStrike, toStrike])
chartTitle = "Total Gamma: $" + str("{:.2f}".format(df['TotalGamma'].sum())) + " Bn per 1% SPX Move, " + dateRange 
plt.title(chartTitle, fontweight="bold", fontsize=12)
plt.xlabel('Strike', fontweight="bold")
plt.ylabel('Spot Gamma Exposure ($ billions/1% move)', fontweight="bold")
plt.axvline(x=spotPrice, color='r', lw=1, label="SPX Spot: " + str("{:,.0f}".format(spotPrice)))
plt.legend()
plt.savefig(os.path.join(outDir,'chart1.png'))
plt.close()
print(f"Chart one saved to {outDir}/chart1.png")


# Chart 2: Absolute Gamma Exposure by Calls and Puts
todayDate,strikes,fromStrike,toStrike,dfAgg,spotPrice,df=conver_csv(tmpFile)
print_header('chart 2: Absolute Gamma Exposure by Calls and Puts')
plt.grid()
plt.bar(strikes, dfAgg['CallGEX'].to_numpy() / 10**9, width=6, linewidth=0.1, edgecolor='k', label="Call Gamma")
plt.bar(strikes, dfAgg['PutGEX'].to_numpy() / 10**9, width=6, linewidth=0.1, edgecolor='k', label="Put Gamma")
plt.xlim([fromStrike, toStrike])
chartTitle = "Total Gamma: $" + str("{:.2f}".format(df['TotalGamma'].sum())) + " Bn per 1% SPX Move, " + dateRange 
plt.title(chartTitle, fontweight="bold", fontsize=12)
plt.xlabel('Strike', fontweight="bold")
plt.ylabel('Spot Gamma Exposure ($ billions/1% move)', fontweight="bold")
plt.axvline(x=spotPrice, color='r', lw=1, label="SPX Spot:" + str("{:,.0f}".format(spotPrice)))
plt.legend()
plt.savefig(os.path.join(outDir,'chart2.png'))
plt.close()
print(f"Chart one saved to {outDir}/chart2.png")



# ---=== CALCULATE GAMMA PROFILE ===---
# parse Data for chart3
todayDate,strikes,fromStrike,toStrike,dfAgg,spotPrice,df=conver_csv(inputFile)

print_header('Chart3: Gama Exposure profile')
levels = np.linspace(fromStrike, toStrike, 60)

# For 0DTE options, I'm setting DTE = 1 day, otherwise they get excluded
df['daysTillExp'] = [1/262 if (np.busday_count(todayDate.date(), x.date())) == 0 \
                           else np.busday_count(todayDate.date(), x.date())/262 for x in df.ExpirationDate]

nextExpiry = df['ExpirationDate'].min()

df['IsThirdFriday'] = [isThirdFriday(x) for x in df.ExpirationDate]
thirdFridays = df.loc[df['IsThirdFriday'] == True]
nextMonthlyExp = thirdFridays['ExpirationDate'].min()

totalGamma = []
totalGammaExNext = []
totalGammaExFri = []

# For each spot level, calc gamma exposure at that point
for level in levels:
    df['callGammaEx'] = df.apply(lambda row : calcGammaEx(level, row['StrikePrice'], row['CallIV'],
                                                          row['daysTillExp'], 0, 0, "call", row['CallOpenInt']), axis = 1)

    df['putGammaEx'] = df.apply(lambda row : calcGammaEx(level, row['StrikePrice'], row['PutIV'],
                                                         row['daysTillExp'], 0, 0, "put", row['PutOpenInt']), axis = 1)

    totalGamma.append(df['callGammaEx'].sum() - df['putGammaEx'].sum())

    exNxt = df.loc[df['ExpirationDate'] != nextExpiry]
    totalGammaExNext.append(exNxt['callGammaEx'].sum() - exNxt['putGammaEx'].sum())

    exFri = df.loc[df['ExpirationDate'] != nextMonthlyExp]
    totalGammaExFri.append(exFri['callGammaEx'].sum() - exFri['putGammaEx'].sum())

totalGamma = np.array(totalGamma) / 10**9
totalGammaExNext = np.array(totalGammaExNext) / 10**9
totalGammaExFri = np.array(totalGammaExFri) / 10**9

# Find Gamma Flip Point
zeroCrossIdx = np.where(np.diff(np.sign(totalGamma)))[0]

negGamma = totalGamma[zeroCrossIdx]
posGamma = totalGamma[zeroCrossIdx+1]
negStrike = levels[zeroCrossIdx]
posStrike = levels[zeroCrossIdx+1]

zeroGamma = posStrike - ((posStrike - negStrike) * posGamma/(posGamma-negGamma))
zeroGamma = zeroGamma[0]

# Chart 3: Gamma Exposure Profile
fig, ax = plt.subplots()
plt.grid()
plt.plot(levels, totalGamma, label="All Expiries")
plt.plot(levels, totalGammaExNext, label="Ex-Next Expiry")
plt.plot(levels, totalGammaExFri, label="Ex-Next Monthly Expiry")
chartTitle = "Gamma Exposure Profile, SPX, " + todayDate.strftime('%d %b %Y')
plt.title(chartTitle, fontweight="bold", fontsize=12)
plt.xlabel('Index Price', fontweight="bold")
plt.ylabel('Gamma Exposure ($ billions/1% move)', fontweight="bold")
plt.axvline(x=spotPrice, color='r', lw=1, label="SPX Spot: " + str("{:,.0f}".format(spotPrice)))
plt.axvline(x=zeroGamma, color='g', lw=1, label="Gamma Flip: " + str("{:,.0f}".format(zeroGamma)))
plt.axhline(y=0, color='grey', lw=1)
plt.xlim([fromStrike, toStrike])
trans = ax.get_xaxis_transform()
plt.fill_between([fromStrike, zeroGamma], min(totalGamma), max(totalGamma), facecolor='red', alpha=0.1, transform=trans)
plt.fill_between([zeroGamma, toStrike], min(totalGamma), max(totalGamma), facecolor='green', alpha=0.1, transform=trans)
plt.legend()
plt.savefig(os.path.join(outDir,'chart3.png'))
plt.close()
print(f"Chart one saved to {outDir}/chart3.png")


### create pdf files
pdf=os.path.join(outDir,'charts.pdf')
with open(pdf, "wb") as f:
    charts= sorted([ os.path.join(outDir,i) for i in os.listdir(outDir) if i.endswith('.png')])
    f.write(img2pdf.convert(charts))
print_header(f"Final output : {pdf}")