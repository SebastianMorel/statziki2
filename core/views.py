import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from django.shortcuts import render
import numpy as np
import math
import statistics
import statsmodels.stats.api as sms
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.compat import lzip
from statsmodels.formula.api import ols
from scipy import stats
from scipy.stats import wilcoxon, pearsonr, mannwhitneyu, norm, hypergeom, binom, poisson, chi2, f
import scipy.integrate as integrate
import pandas as pd
from django.utils.datastructures import MultiValueDictKeyError
import io
import urllib
import base64

def get_request_param(request, param, default_value):
    try:
        return type(default_value)(request.GET[param])
    except (MultiValueDictKeyError, TypeError):
        return default_value

def index(request):
    return render(request, 'index.html')

def hypergeometricPage(request):
    N = get_request_param(request, 'sampleSize', 10)
    n = get_request_param(request, 'triesSize', 3)
    r = get_request_param(request, 'succesfulTries', 5)
    y = get_request_param(request, 'successes', 3)

    rv = hypergeom(N, r, n)
    x = np.arange(0, n+1)

    pmf_hypergeometric = rv.pmf(x)
    p_exakt_hyper = pmf_hypergeometric[y]
    p_less_list_hyper = sum((pmf_hypergeometric[:y]))
    p_less_hyper = (p_less_list_hyper)
    p_more_exact_hyper = 1 - p_less_hyper
    p_more_hyper = 1 - sum((pmf_hypergeometric[:y+1]))
    p_exact_less_hyper = 1 - (p_more_hyper)

    pmf_list = pmf_hypergeometric.tolist()

    theCount = list(range(n))

    pmf_listRounded = [round(i,4) for i in pmf_list]

    print(pmf_listRounded)

    p_exakt_hyper = round(p_exakt_hyper, 5)
    p_less_list_hyper = round(p_less_list_hyper, 5)
    p_less_hyper = round(p_less_hyper, 5)
    p_more_exact_hyper = round(p_more_exact_hyper, 5)
    p_more_hyper = round(p_more_hyper, 5)
    p_exact_less_hyper = round(p_exact_less_hyper, 5)

    context = {'N':N, 'n':n, 'r':r, 'y':y,'p_exakt_hyper':p_exakt_hyper, 'p_less_list_hyper':p_less_list_hyper, 'p_less_hyper':p_less_hyper, 'p_more_exact_hyper':p_more_exact_hyper,'p_more_hyper':p_more_hyper,'p_exact_less_hyper':p_exact_less_hyper,
    'pmf_listRounded':pmf_listRounded, 'theCount':theCount}
    return render(request, 'hypergeometricPage.html', context)

def Poisson(request):
    mu = get_request_param(request, 'mu', 7)
    limit = get_request_param(request, 'limit', 3)

    mean, var, skew, kurt = poisson.stats(mu, moments='mvsk')
    x = np.arange(poisson.ppf(0.00001, mu), poisson.ppf(0.99999, mu))

    x = np.arange(poisson.ppf(0.0001, mu), poisson.ppf(0.9999, mu))
    allprobs = poisson.pmf(x, mu)
    probexakt = allprobs[limit]
    probunder = sum((allprobs[:limit]))
    probunder_equal = sum((allprobs[:limit+1]))
    probover = 1 - probunder_equal
    probover_equal = 1 - probunder

    variance = mu**2
    indexPrb = list(range(len(allprobs)))
    allprobs = allprobs.tolist()

    probexakt = round(probexakt, 4)
    probunder = round(probunder, 4)
    probunder_equal = round(probunder_equal, 4)
    probover = round(probover, 4)
    probover_equal = round(probover_equal, 4)
    variance = round(variance, 4)

    allprobsRounded = [round(i,4) for i in allprobs]

    context = {'mu':mu, 'limit':limit, 'probexakt':probexakt, 'probunder':probunder, 'probunder_equal':probunder_equal, 'probover':probover, 'probover_equal':probover_equal,'variance':variance,
    'indexPrb':indexPrb, 'allprobsRounded':allprobsRounded}
    return render(request, 'Poisson.html', context)

def draw_z_score(x, cond, mu, sigma, title):
    y = norm.pdf(x, mu, sigma)
    z = x[cond]
    plt.plot(x, y, color='black')
    plt.fill_between(z, 0, norm.pdf(z, mu, sigma), color='#539ecd')
    plt.title(title)

def Normal(request):
    Xval = get_request_param(request, 'Xval', 400)
    MU = get_request_param(request, 'MU', 450)
    STD = get_request_param(request, 'STD', 20)

    Z_Less_Than = (Xval - MU)/STD
    x1 = np.arange(-3,3,0.001)
    draw_z_score(x1, x1<Z_Less_Than, 0, 1, 'z<' + str(Z_Less_Than))
    
    fig1 = plt.gcf()
    buf1 = io.BytesIO()
    fig1.savefig(buf1,format='png')
    buf1.seek(0)
    string1 = base64.b64encode(buf1.read())
    uri1 = urllib.parse.quote(string1)
    plt.clf()

    Z_More_Than = (Xval - MU)/(STD)
    x = np.arange(-3,3,0.001)
    draw_z_score(x, x > Z_More_Than, 0, 1, ' z>' + str(Z_More_Than))

    fig = plt.gcf()
    buf = io.BytesIO()
    fig.savefig(buf,format='png')
    buf.seek(0)
    string = base64.b64encode(buf.read())
    uri = urllib.parse.quote(string)
    plt.clf()

    lessOrEqualto = norm.cdf(Xval, MU, STD) #cdf x < val
    moreOrEqualto = 1-norm.cdf(Xval, MU, STD) #cdf val > x

    lessOrEqualto = round(lessOrEqualto,5)
    moreOrEqualto = round(moreOrEqualto,5)
    Z_Less_Than = round(Z_Less_Than,5)
    Z_More_Than = round(Z_More_Than,5)

    context = {'lessOrEqualto':lessOrEqualto, 'moreOrEqualto':moreOrEqualto, 'data':uri, 'data1': uri1,
    'Xval':Xval, 'MU':MU, 'STD':STD,'Z_Less_Than':Z_Less_Than,'Z_More_Than':Z_More_Than}
    return render(request, 'Normal.html', context)

def Normalinterval(request):
    lowerX = get_request_param(request, 'lowerX', 0.12)
    upperX = get_request_param(request, 'upperX', 0.14)
    MU = get_request_param(request, 'MU', 0.13)
    STD = get_request_param(request, 'STD', 0.005)

    upperZ = (upperX - MU)/STD
    lowerZ = (lowerX - MU)/STD

    upperZ = round(upperZ, 5)
    lowerZ = round(lowerZ, 5)

    intervalProbability = norm.cdf(upperX, MU, STD)-norm.cdf(lowerX, MU, STD) #Interval probability

    x = np.arange(-3,3,0.001)
    draw_z_score(x, (lowerZ < x) & (x < upperZ), 0, 1, str(lowerZ)+'<Z<'+str(upperZ))

    fig = plt.gcf()
    buf = io.BytesIO()
    fig.savefig(buf,format='png')
    buf.seek(0)
    string = base64.b64encode(buf.read())
    uri = urllib.parse.quote(string)
    plt.clf()

    intervalProbability = round(intervalProbability,5)
    upperZ = round(upperZ,5)
    lowerZ = round(lowerZ,5)

    context = {'intervalProbability': intervalProbability, 'data': uri, 'lowerX':lowerX, 'upperX':upperX,
    'MU':MU, 'STD':STD,'upperZ':upperZ,'lowerZ':lowerZ}
    return render(request, 'Normalinterval.html', context)

def Binomial(request):
    n = get_request_param(request, 'n', 20)
    p = get_request_param(request, 'p', 0.8)
    X = get_request_param(request, 'X', 14)

    mean, var, skew, kurt = binom.stats(n, p, moments='mvsk')
    x = np.arange(binom.ppf(0.0000000000000000000000000000000000001, n, p), binom.ppf(0.9999999999999999999999999999999999999, n, p))

    p_EXAKT = float(binom.pmf(X, n, p)) 
    P_lessEqualX = float(binom.cdf(X, n, p))
    P_moreX = 1 - P_lessEqualX
    p_lessX = float(binom.cdf(X-1, n, p))
    p_moreEqualX = float(1 - p_lessX)

    prb = binom.pmf(x, n, p) #Alla probabilites
    indexPrb = list(range(len(prb)))

    problist = prb.tolist()
    
    p_EXAKT= round(p_EXAKT,5)
    P_lessEqualX= round(P_lessEqualX,5)
    P_moreX= round(P_moreX,5)
    p_lessX= round(p_lessX,5)
    p_moreEqualX= round(p_moreEqualX,5)

    problist = [round(i,4) for i in problist]

    context = {'p_EXAKT': p_EXAKT, 'P_lessEqualX': P_lessEqualX, 'P_moreX': P_moreX, 'p_lessX': p_lessX,
    'p_moreEqualX': p_moreEqualX, 'indexPrb':indexPrb,'problist':problist, 'n':n, 'p':p, 'X':X}
    return render(request, 'Binomial.html', context)

def mulRegression(request):
    Yinput = get_request_param(request, 'Yinput', '1\n1\n1')
    X1input = get_request_param(request, 'X1input', '2\n2\n2')
    X2input = get_request_param(request, 'X2input', '3\n3\n3')

    try:
        Y = [int(x) for x in Yinput.split()]
        X1 = [int(x) for x in X1input.split()]
        X2 = [int(x) for x in X2input.split()]
    except ValueError:
        Y = [float(x) for x in Yinput.split()]
        X1 = [float(x) for x in X1input.split()]
        X2 = [float(x) for x in X2input.split()]
    
    if len(Y) != len(X1):
        lenY = int(len(Y))
        lenX1 = int(len(X1))
        if lenY > lenX1:
            for i in range(lenY-lenX1):
                X1.append(0)
        if lenX1 > lenY:
            for i in range(lenX1-lenY):
                Y.append(0)

    def listToString(s):
        str1 = ""  
        for ele in s:  
            str1 += str(ele)
            str1 += '\n'
        return str1

    Yinput = listToString(Y)
    Xinput = listToString(X1)

    n = len(Y)

    data = pd.DataFrame({"X1": X1, "X2": X2, "Y":Y})
    Xvalues = pd.DataFrame({"X1": X1, "X2": X2})
    model = ols("Y ~ X1 + X2", data).fit()
    AnovaModel = pd.DataFrame((anova_lm(model)))

    AnovaModel_Table = AnovaModel.iloc[:,0:8]
    AnovaModel_StylishTable = AnovaModel_Table.to_html(classes='table-sm table-hover table-striped') 

    #Coefficients
    Coefficients = model.params.tolist()

    Coef1 = Coefficients[0] 
    Coef2 = Coefficients[1] 
    Coef3 = Coefficients[2] 
    #Correlation
    vif = pd.DataFrame()
    vif["VIF Factor"] = [variance_inflation_factor(Xvalues.values, i) for i in range(Xvalues.shape[1])]
    vif["features"] = Xvalues.columns

    data_corr = {'A': Y,
            'B': X1,
            'C': X2
            }
    corrDF = pd.DataFrame(data_corr, columns=['A', 'B', 'C'])
    corrMatrix = corrDF.corr()

    corrMatrix_Table = corrMatrix.iloc[:,:]
    corrMatrix_StylishTable = corrMatrix_Table.to_html(classes='table-sm table-hover table-striped')

    vif_Table = vif.iloc[:,:]
    vif_StylishTable = vif_Table.to_html(classes='table-sm table-hover table-striped')
    #R^2
    sumSQcol = AnovaModel.iloc[:,1].values
    SSR = sumSQcol[0] + sumSQcol[1]
    SST = SSR + sumSQcol[2]
    Rsquared = SSR/SST
    adjRsquared = 1 - ((1-Rsquared) * (n-1) / (n-2-1))

    #Resid
    Residuals = model.resid
    Predicted = model.fittedvalues
    Residual_list = Residuals.tolist()
    Predicted_values = Predicted.tolist()

    def merge(Predicted_values, Residual_list):
        merged_list = [[Predicted_values[i], Residual_list[i]] for i in range(0, len(Predicted_values))]
        return merged_list

    plotResPred = merge(Predicted_values, Residual_list)

    #F test residualer konstant varians
    name = ['Lagrange multiplier statistic', 'p-value', 
            'f-value', 'f p-value']
    test = sms.het_breuschpagan(model.resid, model.model.exog)
    lzip(name, test)

    #Normality test
    name = ['Jarque-Bera', 'Chi^2 two-tail prob.', 'Skew', 'Kurtosis']
    test = sms.jarque_bera(model.resid)
    lzip(name, test)
    stats.probplot(model.resid, dist="norm", plot= plt)
    plt.title("Model1 Residuals Q-Q Plot")
    plt.clf()

    Residual_list.sort()
    rangeList = list(range(1, n+1))

    def mergeQQ(Zquantiles, Residual_list):
        mergedQQ_list = [[Zquantiles[i], Residual_list[i]] for i in range(0, len(Zquantiles))]
        return mergedQQ_list

    QQvalues = []
    for number in rangeList:
      QQvalues.append((number - 0.5)/n)

    Zquantiles =[]
    for number in QQvalues:
      Zquantiles.append(norm.ppf(number))

    plotQQ = mergeQQ(Zquantiles,Residual_list)

    sumY1 = sum(Y)
    sumX1 = sum(X1)
    sumX2 = sum(X2)
    Ysq = [number ** 2 for number in Y]
    sumYsq = sum(Ysq)
    X1sq = [number ** 2 for number in X1]
    sumX1sq = sum(X1sq)
    X2sq = [number ** 2 for number in X2]
    sumX2sq = sum(X2sq)
    X1Y = []

    for num1, num2 in zip(X1, Y):
    	X1Y.append(num1 * num2)
    
    X2Y = []
    for num1, num2 in zip(X2, Y):
    	X2Y.append(num1 * num2)

    X1X2 = []
    for num1, num2 in zip(X1, X2):
	    X1X2.append(num1 * num2)

    X1Y = sum(X1Y)
    X2Y = sum(X2Y)
    X1X2 = sum(X1X2)

    n = len(Y)

    FirstRow = X1Y - (sumX1 * sumY1 / n)
    SecondRow = X2Y - (sumX2 * sumY1 / n)
    ThirdRow = X1X2 - (sumX1 * sumX2 / n)
    FourthRow = sumX1sq - (sumX1**2 / n)
    FifthRow = sumX2sq - (sumX2**2 / n)

    mean_Y = statistics.mean(Y)
    mean_X1 = statistics.mean(X1)
    mean_X2 = statistics.mean(X2)

    mean_Y = round(mean_Y, 3)
    mean_X1 = round(mean_X1, 3)
    mean_X2 = round(mean_X2, 3)
    Coef1 = round(Coef1, 5)
    Coef2 = round(Coef2, 5)
    Coef3 = round(Coef3, 5)
    FirstRow = round(FirstRow, 3)
    SecondRow = round(SecondRow, 3)
    ThirdRow = round(ThirdRow, 3)
    FourthRow = round(FourthRow, 3)
    FifthRow = round(FifthRow, 3)

    context = {'plotQQ': plotQQ, 'plotResPred':plotResPred, 'vif_StylishTable':vif_StylishTable, 'Coef1':Coef1, 'Coef2':Coef2,
    'Coef3':Coef3, 'AnovaModel_StylishTable':AnovaModel_StylishTable, 'corrMatrix_StylishTable':corrMatrix_StylishTable, 'Yinput': Yinput, 'X1input':X1input,
    'X2input':X2input, 'Rsquared':Rsquared, 'adjRsquared':adjRsquared, 'sumY1':sumY1,'sumX1':sumX1,'sumX2':sumX2,'Ysq':Ysq,'sumYsq':sumYsq, 'X1sq':X1sq,'sumX1sq':sumX1sq,
    'X2sq':X2sq, 'sumX2sq':sumX2sq,'X1Y':X1Y, 'X2Y':X2Y,'X1X2':X1X2, 'FirstRow':FirstRow, 'SecondRow':SecondRow,'ThirdRow':ThirdRow, 'FourthRow':FourthRow, 'FifthRow':FifthRow,'mean_Y':mean_Y,
    'mean_X1':mean_X1, 'mean_X2':mean_X2}
    return render(request, 'multiRegression.html', context)

def simpleRegression(request):
    Yinput = get_request_param(request, 'Yinput', '20\n25\n30\n37\n45\n47\n55\n59\n62\n65')
    Xinput = get_request_param(request, 'Xinput', '15\n17\n25\n32\n51\n43\n60\n65\n58\n68')

    try:
        Y = [int(x) for x in Yinput.split()]
        X1 = [int(x) for x in Xinput.split()]
    except ValueError:
        Y = [float(x) for x in Yinput.split()]
        X1 = [float(x) for x in Xinput.split()]

    if len(Y) != len(X1):
        lenY = int(len(Y))
        lenX1 = int(len(X1))
        if lenY > lenX1:
            for i in range(lenY-lenX1):
                X1.append(0)
        if lenX1 > lenY:
            for i in range(lenX1-lenY):
                Y.append(0)

    def listToString(s):
        str1 = ""  
        for ele in s:  
            str1 += str(ele)
            str1 += '\n'
        return str1

    Yinput = listToString(Y)
    Xinput = listToString(X1)

    n = len(Y)
    XY = []
    for num1, num2 in zip(Y, X1):
      XY.append(num1*num2)

    #Coefficients
    SUM_XY = sum(XY)
    SUM_X = sum(X1)
    Xsquared = [number**2 for number in X1]
    Ysquared = [number**2 for number in Y]
    SUM_Ysquared = sum(Ysquared)
    SUM_Xsquared = sum(Xsquared)
    SUM_Y = sum(Y)
        
    SUM_Xsquare = SUM_X**2
    b = (SUM_XY - SUM_X * SUM_Y/n) / (SUM_Xsquared - (SUM_X**2) /n)

    Y_mean = statistics.mean(Y)
    X_man = statistics.mean(X1)

    a = Y_mean - b * X_man

    #Anova
    data = pd.DataFrame({"X1": X1, "Y":Y})
    model = ols("Y ~ X1", data).fit()
    AnovaModel = pd.DataFrame((anova_lm(model)))
    AnovaModel_Table = AnovaModel.iloc[:,0:8]
    AnovaModel_StylishTable = AnovaModel_Table.to_html(classes='table-sm table-striped table-hover') 

    Predicted = model.fittedvalues
    Predicted_list = Predicted.tolist()

    Error = [value - Y_mean for value in Predicted_list]
    SSR = sum([value**2 for value in Error])
    SST = [value - Y_mean for value in Y]
    SST_squared = float(sum([value**2 for value in SST]))
    SSE = SST_squared-SSR
    MSE = SSE / (n-2)
    Fvalue_simple = SSR/MSE
    y1 = Y[0]
    y2 =Y[1]
    yn = Y[-1]
    #fittedline
    x = np.array(X1)
    y = np.array(Y)
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    plt.plot(X1, Y, 'o', label='original data')
    plt.plot(X1, intercept + slope*x, 'r', label='fitted line')
    plt.legend()
    plt.show()

    fig = plt.gcf()
    buf = io.BytesIO()
    fig.savefig(buf,format='png')
    buf.seek(0)
    string = base64.b64encode(buf.read())
    uri = urllib.parse.quote(string)
    plt.clf()

    df1_simple = 1
    df2_simple = n-2
    df3_simple = n-1
    p_value_simple = f.cdf(Fvalue_simple, df1_simple, df2_simple)
    a = round(a, 4)
    b = round(b, 4)
    Y_mean = round(Y_mean, 4)
    X_man = round(X_man, 4)
    SST = sum(SST)
    SST_squared = round(SST_squared, 4)
    SSE = round(SSE, 4)
    SSR = round(SSR, 4)
    MSE = round(MSE, 4)
    Fvalue_simple = round(Fvalue_simple, 4)
    p_value_simple = round(p_value_simple, 4)
    p_value_simple = 1 - p_value_simple
    p_value_simple = round(p_value_simple, 4)

    context = {'Yinput':Yinput, 'Xinput':Xinput,'AnovaModel_StylishTable':AnovaModel_StylishTable, 'SUM_XY':SUM_XY, 'SUM_X':SUM_X, 'SUM_Y':SUM_Y, 'n':n,
    'SUM_Xsquared':SUM_Xsquared,'SUM_Xsquare':SUM_Xsquare, 'b':b, 'Y_mean':Y_mean, 'X_man':X_man, 'y1':y1,'y2':y2,'yn':yn, 'a':a,'SST_squared':SST_squared,'SSR':SSR,'SSE':SSE,'SUM_Ysquared':SUM_Ysquared,'MSE':MSE,'Fvalue_simple':Fvalue_simple,'df1_simple':df1_simple,'df2_simple':df2_simple,
    'df3_simple':df3_simple,'p_value_simple':p_value_simple, 'uri':uri}
    return render(request, 'simpleRegression.html', context)

def Anova(request):
    T1 = get_request_param(request, 'T1', '24\n28\n37\n30')
    T2 = get_request_param(request, 'T2', '37\n44\n31\n35')
    T3 = get_request_param(request, 'T3', '42\n47\n52\n35')
    T4 = get_request_param(request, 'T4', '46\n43\n57\n34')
    T5 = get_request_param(request, 'T5', '58\n46\n52\n38')

    try:
        X0 = [int(x) for x in T1.split()]
        X1 = [int(x) for x in T2.split()]
        X2 = [int(x) for x in T3.split()]
        X3 = [int(x) for x in T4.split()]
        X4 = [int(x) for x in T5.split()]
    except ValueError:
        X0 = [float(x) for x in T1.split()]
        X1 = [float(x) for x in T2.split()]
        X2 = [float(x) for x in T3.split()]
        X3 = [float(x) for x in T4.split()]
        X4 = [float(x) for x in T5.split()]        

    N = len(X0 + X1 + X2 + X3 + X4)
    SumOfAll = sum(X0+X1+X2+X3+X4)

    n0 = len(X0)
    n1 = len(X1)
    n2 = len(X2)
    n3 = len(X3)
    n4 = len(X4)

    X0sq = [number1 ** 2 for number1 in X0]
    X1sq = [number2 ** 2 for number2 in X1]
    X2sq = [number3 ** 2 for number3 in X2]
    X3sq = [number4 ** 2 for number4 in X3]
    X4sq = [number5 ** 2 for number5 in X4]

    totX0sq = sum(X0sq)
    totX1sq = sum(X1sq)
    totX2sq = sum(X2sq)
    totX3sq = sum(X3sq)
    totX4sq = sum(X4sq)

    TotalSquared = totX0sq + totX1sq + totX2sq + totX3sq + totX4sq

    #square varje nivå SStreatments
    X0tot = sum(X0)
    X1tot = sum(X1)
    X2tot = sum(X2)
    X3tot = sum(X3)
    X4tot = sum(X4)

    X0totSQ_table = X0tot**2
    X1totSQ_table = X1tot**2
    X2totSQ_table = X2tot**2
    X3totSQ_table = X3tot**2
    X4totSQ_table = X4tot**2

    X0totSQ = X0tot**2 / n0
    X1totSQ = X1tot**2 / n1
    X2totSQ = X2tot**2 / n2

    try:
        X3totSQ = X3tot**2 / n3
    except ZeroDivisionError:
        X3totSQ = 0
    try:
        X4totSQ = X4tot**2 / n4
    except ZeroDivisionError:
        X4totSQ = 0

    sumYsq = X0totSQ + X1totSQ + X2totSQ + X3totSQ + X4totSQ
    SStreatment = sumYsq - ((SumOfAll)**2/N)

    SST = TotalSquared - ((SumOfAll)**2/N)
    #SSE
    SSE = SST - SStreatment

    k = 5
    if len(X4) == 0:
      k = 4 

    if len(X3) == 0:
      k = 3

    MStreatment = SStreatment / (k-1)
    MSE = SSE / (N-k)
    df3 = N-1
    df2 = N-k
    df1 = k-1

    Fvalue = MStreatment / MSE
    single_tailed_pval = 1 - (f.cdf(Fvalue,df1,df2))

    df1 = round(df1,4)
    df2 = round(df2,4)
    df3 = round(df3,4)
    MSE = round(MSE,4)
    MStreatment = round(MStreatment,4)
    single_tailed_pval = round(single_tailed_pval,4)
    SST = round(SST,4)
    SStreatment = round(SStreatment,4)
    SSE = round(SSE,4)
    Fvalue = round(Fvalue,4)
    X0totSQ_table = round(X0totSQ_table,4)
    X1totSQ_table = round(X1totSQ_table,4)
    X2totSQ_table = round(X2totSQ_table,4)
    X3totSQ_table = round(X3totSQ_table,4)
    X4totSQ_table = round(X4totSQ_table,4)
    SumOfAll = round(SumOfAll,4)
    TotalSquared = round(TotalSquared,4)
    n0 = round(n0,4)
    n1 = round(n1,4)
    n2 = round(n2,4)
    n3 = round(n3,4)
    N = round(N,4)

    context = {'df1':df1,'df2':df2,'df3':df3,'MSE':MSE, 'MStreatment':MStreatment,'single_tailed_pval':single_tailed_pval, 'SST':SST,
    'T1':T1, 'T2':T2, 'T3':T3, 'T4':T4, 'T5':T5, 'SStreatment':SStreatment, 'SSE':SSE,'Fvalue':Fvalue,'X0totSQ_table':X0totSQ_table,'X1totSQ_table':X1totSQ_table,
    'X2totSQ_table':X2totSQ_table, 'X3totSQ_table':X3totSQ_table,'X4totSQ_table':X4totSQ_table,'SumOfAll':SumOfAll,'TotalSquared':TotalSquared,'n0':n0,
    'n1':n1,'n2':n2, 'n3':n3,'N':N}
    return render(request, 'Anova.html', context)

def expProb(request):
    Lambda = get_request_param(request, 'Lambda', 5)
    LX = get_request_param(request, 'LX', 0.2)
    UX = get_request_param(request, 'UX', 0.5)

    B = 1/Lambda
    f = lambda x:(1/B)*2.71828182845904**(-x/B)
    i = integrate.quad(f, LX, UX)
    pd_i = pd.DataFrame(i)
    pd_i
    less_prb_exp = pd_i.iloc[0,0] #Värdet på integral #Sannolikhet att få under 1
    greater_prb_exp = 1 - less_prb_exp #Sannolikhet att få över 1

    greater_prb_exp = round(greater_prb_exp,5)
    less_prb_exp = round(less_prb_exp,5)
    B = round(B,5)

    context = {'Lambda':Lambda, 'UX':UX, 'LX':LX, 'greater_prb_exp':greater_prb_exp,'less_prb_exp':less_prb_exp, 'B':B}
    return render(request, 'expProb.html', context)

def expProb1(request):
    B = get_request_param(request, 'B', 2.4)
    LX = get_request_param(request, 'LX', 0)
    UX = get_request_param(request, 'UX', 3)

    f = lambda x:(1/B)*2.71828182845904**(-x/B)
    i = integrate.quad(f, LX, UX)
    pd_i = pd.DataFrame(i)
    pd_i
    less_prb_exp = pd_i.iloc[0,0] #Värdet på integral #Sannolikhet att få under 1
    greater_prb_exp = 1 - less_prb_exp #Sannolikhet att få över 1

    greater_prb_exp = round(greater_prb_exp,5)
    less_prb_exp = round(less_prb_exp,5)
    B = round(B,5)

    context = {'UX':UX, 'LX':LX, 'greater_prb_exp':greater_prb_exp,'less_prb_exp':less_prb_exp, 'B':B}
    return render(request, 'expProb1.html', context)

def Ttest2independant(request):
    Sample1 = get_request_param(request, 'Sample1', '96.9 97.4 97.5 97.8 97.8 97.9 98 98.6 98.8')
    Sample2 = get_request_param(request, 'Sample2', '97.8 98 98.2 98.2 98.2 98.6 98.8 99.2 99.4')

    try:
        Observation1 = [int(x) for x in Sample1.split()]
        Observation2 = [int(x) for x in Sample2.split()]
    except ValueError:
        Observation1 = [float(x) for x in Sample1.split()]
        Observation2 = [float(x) for x in Sample2.split()]

    variance1 = statistics.variance(Observation1)
    variance2 = statistics.variance(Observation2)
    mean1 = statistics.mean(Observation1)
    mean2 = statistics.mean(Observation2)
    n1 = len(Observation1)
    n2 = len(Observation2)

    PooledSTD = ((n1-1)*variance1 + (n2-1)*variance2)/(n1+n2-2)

    tvalue = (mean1-mean2)/((PooledSTD * (1/n1 + 1/n2))**0.5)
    degressFree = n1+n2-2

    pval = stats.t.sf(np.abs(tvalue), n1+n2-2)*2  # two-sided pvalue = Prob(abs(t)>tt)

    variance1 = round(variance1, 4)
    variance2 = round(variance2, 4)
    PooledSTD = round(PooledSTD, 4)
    tvalue = round(tvalue, 4)
    pval = round(pval,4)
    n1 = round(n1,4)
    n2 = round(n2,4)
    mean1 = round(mean1,4)
    mean2 = round(mean2,4)

    context = {'PooledSTD':PooledSTD,'tvalue':tvalue,'pval':pval,'Sample1':Sample1, 'Sample2':Sample2,'n1':n1,'n2':n2,'variance1':variance1,
    'variance2':variance2, 'mean1':mean1, 'mean2':mean2}
    return render(request, 'Ttest2independant.html', context)

def PairedTtest(request):
    Sample1 = get_request_param(request, 'Treatment1', '70 68 66 98 77 66 55 3 98')
    Sample2 = get_request_param(request, 'Treatment2', '73 69 78 99 88 68 78 4 99')

    try:
        Observation1 = [int(x) for x in Sample1.split()]
        Observation2 = [int(x) for x in Sample2.split()]
    except ValueError:
        Observation1 = [float(x) for x in Sample1.split()]
        Observation2 = [float(x) for x in Sample2.split()]

    n = len(Observation1)
    Diff = []
    zip_object = zip(Observation1, Observation2)
    for list1_i, list2_i in zip_object:
        Diff.append(list1_i-list2_i)

    DiffSTD = statistics.stdev(Diff)
    DiffMEAN = statistics.mean(Diff)

    squared_diff = [number ** 2 for number in Diff]
    sumsquaredDiff = sum(squared_diff)
    sumDiff = sum(Diff)

    tvalue = DiffMEAN / (DiffSTD / n**0.5)
    pval_two_sided_paired = stats.t.sf(np.abs(tvalue), n-1)*2 #twosided p val
    pval_one_sided_paired = stats.t.sf(np.abs(tvalue), n-1) #one sided p val

    tvalue = round(tvalue,4)
    pval_two_sided_paired = round(pval_two_sided_paired,4)
    pval_one_sided_paired = round(pval_one_sided_paired,4)
    sumsquaredDiff = round(sumsquaredDiff,4)
    sumDiff = round(sumDiff,4)
    n = round(n,4)
    DiffSTD = round(DiffSTD,4)
    DiffMEAN = round(DiffMEAN,4)

    context = {'tvalue':tvalue,'Sample1':Sample1,'Sample2':Sample2,'pval_two_sided_paired':pval_two_sided_paired,
    'pval_one_sided_paired':pval_one_sided_paired,'sumsquaredDiff':sumsquaredDiff,'sumDiff':sumDiff,'n':n,'DiffSTD':DiffSTD,'DiffMEAN':DiffMEAN}
    return render(request, 'PairedTtest.html', context)

def varianceAnalysis(request):
    Significance = get_request_param(request, 'Significance', 1)
    Variance = get_request_param(request, 'Variance', 1)
    n = get_request_param(request, 'n', 1)

    a = Significance / 2
    UpperChi = chi2.ppf(a, n-1)
    LowerChi = chi2.ppf(1-a, n-1)

    LowerInterval = ((n-1)*Variance) / LowerChi
    UpperInterval = ((n-1)*Variance) / UpperChi

    LowerInterval = round(LowerInterval, 4)
    UpperInterval = round(UpperInterval, 4)

    LowerChi = round(LowerChi, 4)
    UpperChi = round(UpperChi, 4)

    context = {'Significance':Significance,'Variance':Variance,'n':n,'LowerInterval':LowerInterval,'UpperInterval':UpperInterval, 'LowerChi':LowerChi,'UpperChi':UpperChi}
    return render(request,'varianceAnalysis.html',context)

def varianceAnalysis2(request):
    Var1 = get_request_param(request, 'Var1', 1)
    n1 = get_request_param(request, 'n1', 1)
    Var2 = get_request_param(request, 'Var2', 1)
    n2 = get_request_param(request, 'n2', 1)
    SignificanceF = get_request_param(request, 'SignificanceF', 1)

    F = Var1/Var2
    p_value = f.cdf(F, n1-1, n2-1)
    p_value = float(p_value)
    twotailPval = p_value*2

    F = round(F,4)
    twotailPval = round(twotailPval,4)
    SignificanceF = round(SignificanceF,4)
    p_value = round(p_value,4)

    context = {'F':F,'twotailPval':twotailPval,'Var1':Var1,'Var2':Var2,'n1':n1,'n2':n2,
    'SignificanceF':SignificanceF,'p_value':p_value}
    return render(request,'varianceAnalysis2.html',context)

def ProportionTest(request):
    p1 = get_request_param(request, 'p1', 0.5)
    p2 = get_request_param(request, 'p2', 0.6)
    n = get_request_param(request, 'n', 1)

    Z = (p1-p2)/((p1+p2)/n)**0.5
    p_values_one_sample = norm.sf(abs(Z))

    Z = round(Z,5)
    p_values_one_sample = round(p_values_one_sample,5)

    context = {'Z':Z,'p1':p1,'p2':p2,'n':n,'p_values_one_sample':p_values_one_sample}
    return render(request, 'ProportionTest.html', context)

def ProportionTest2(request):
    p1_two = get_request_param(request, 'p1_two', 0.5)
    p2_two = get_request_param(request, 'p2_two', 0.6)
    n1 = get_request_param(request, 'n1', 10)
    n2 = get_request_param(request, 'n2', 2)

    P = (n1*p1_two + n2*p2_two)/(n1+n2)
    Z_two = (p1_two-p2_two)/(P*(1-P)*(1/n1 + 1/n2))**0.5
    p_values_two = norm.sf(abs(Z_two))

    Z_two = round(Z_two,5)
    p_values_two = round(p_values_two,5)
    P = round(P,5)

    context = {'Z_two':Z_two,'p1_two':p1_two,'p2_two':p2_two,'n1':n1,'n2':n2,'p_values_two':p_values_two,'P':P}
    return render(request, 'ProportionTest2.html', context)

def zTest(request):
    Sample_1List = get_request_param(request, 'Sample_1', 1.65)
    Variance_1 = get_request_param(request, 'Variance1', 0.0676)
    n1 = get_request_param(request, 'n1', 30)
    Sample_2List = get_request_param(request, 'Sample_2', 1.43)
    Variance_2 = get_request_param(request, 'Variance2', 0.0484)
    n2 = get_request_param(request, 'n2', 35)

    Z_STD_known = (Sample_1List - Sample_2List)/(Variance_1/n1 + Variance_2/n2)**0.5
    p_values_one_stdknown = norm.sf(abs(Z_STD_known)) #one-sided
    p_values_two_stdknown = norm.sf(abs(Z_STD_known))*2 #twoside

    p_values_two_stdknown = round(p_values_two_stdknown,5)
    p_values_one_stdknown = round(p_values_one_stdknown,5)
    Z_STD_known = round(Z_STD_known,5)

    p_values_one_stdknown = norm.sf(abs(Z_STD_known)) #one-sided
    p_values_two_stdknown = norm.sf(abs(Z_STD_known))*2 #twoside

    p_values_one_stdknown = round(p_values_one_stdknown, 5)
    p_values_two_stdknown = round(p_values_two_stdknown, 5)

    context = {'p_values_two_stdknown':p_values_two_stdknown,'p_values_one_stdknown':p_values_one_stdknown,
    'Sample_1List':Sample_1List,'Variance_1':Variance_1,'n1':n1,'n2':n2,'Sample_2List':Sample_2List,'Variance_2':Variance_2,
    'Z_STD_known':Z_STD_known,'p_values_one_stdknown':p_values_one_stdknown,'p_values_two_stdknown':p_values_two_stdknown}

    return render(request, 'zTest.html', context)

def zTest2(request):
    Sample = get_request_param(request, 'Sample', 207)
    mu = get_request_param(request, 'mu', 210)
    variance1 = get_request_param(request, 'variance1', 100)
    n = get_request_param(request, 'nSample', 60)

    sigma = variance1**0.5

    Z = (Sample-mu)/(sigma/n**0.5)

    Z = round(Z,5)
    sigma = round(sigma,5)

    p_values_one_stdknown = norm.sf(abs(Z)) #one-sided
    p_values_two_stdknown = norm.sf(abs(Z))*2 #twoside

    p_values_one_stdknown = round(p_values_one_stdknown, 5)
    p_values_two_stdknown = round(p_values_two_stdknown, 5)

    context = {'Z':Z,'mu':mu,'n':n,'Sample':Sample,'variance1':variance1, 'sigma':sigma,'p_values_one_stdknown':p_values_one_stdknown,'p_values_two_stdknown':p_values_two_stdknown}

    return render(request, 'zTest2.html', context)

def riskRatio(request):
    Group11 = get_request_param(request, 'Group11', 139)
    Group12 = get_request_param(request, 'Group12', 239)
    Group21 = get_request_param(request, 'Group21', 10898)
    Group22 = get_request_param(request, 'Group22', 10795)

    r1 = Group11 + Group12
    r2 = Group21 + Group22
    k1 = Group11 + Group21
    k2 = Group12 + Group22
    RR = (Group11 / (Group11 + Group21)) / (Group12 / (Group12 + Group22))
    n = r1 + r2

    #P-value risk
    ChiRisk = (((Group11 * Group22 - Group12 * Group21)**2) / ((Group11+Group12) * (Group21 + Group22) * (Group11 + Group21) * (Group12 + Group22))) * (Group11+Group21+Group22+Group12)
    pvalueRisk = round(stats.chi2.pdf(ChiRisk , 1),4)
    lnRR = np.log(RR)

    seRR = (1/Group11 - 1/(Group11+Group21) + 1/Group12 - 1/(Group12 + Group22))**0.5

    #Confidence interval RR
    LowerlnRR = lnRR - 1.96*seRR
    UpperlnRR = lnRR + 1.96*seRR

    LowerRR = math.exp(LowerlnRR)
    UpperRR = math.exp(UpperlnRR)

    RR = round(RR,5)
    ChiRisk = round(ChiRisk,5)
    pvalueRisk = round(pvalueRisk,5)
    seRR = round(seRR,5)
    LowerlnRR = round(LowerlnRR,5)
    UpperlnRR = round(UpperlnRR,5)
    LowerRR = round(LowerRR,5)
    UpperRR = round(UpperRR,5)
    r1 = round(r1,5)
    r2 = round(r2,5)
    k1 = round(k1,5)
    k2 = round(k2,5)
    n = round(n,5)
    lnRR = round(lnRR,5)

    context = {'RR':RR,'ChiRisk':ChiRisk,'pvalueRisk':pvalueRisk,'seRR':seRR,'LowerlnRR':LowerlnRR,'UpperlnRR':UpperlnRR,
    'LowerRR':LowerRR,'UpperRR':UpperRR,'Group11':Group11,'Group12':Group12,'Group21':Group21,'Group22':Group22,'r1':r1,'r2':r2,'k1':k1,'k2':k2,'n':n,'lnRR':lnRR}
    return render(request, 'riskRatio.html', context)

def oddsRatio(request):
    a = get_request_param(request, 'a', 65)
    b = get_request_param(request, 'b', 12)
    c = get_request_param(request, 'c', 846)
    d = get_request_param(request, 'd', 42)

    totalab = a+b
    totalcd = c+d
    totalac = a+c
    totalbd = b+d
    total = totalab+totalcd

    OA = a/b
    OB = c/d
    OR = OA/OB
    
    SElnOR = (1/a + 1/b + 1/c + 1/d)**0.5
    lnOR = np.log(OR)
    LowerlnOR = OR - 1.96*SElnOR
    UpperlnOR = OR + 1.96*SElnOR
    LowerlnOR = float(LowerlnOR)
    UpperlnOR = float(UpperlnOR)
    LowerOR = math.exp(LowerlnOR)
    UpperOR = math.exp(UpperlnOR)

    OA = round(OA,5)
    OB = round(OB,5)
    OR = round(OR,5)
    SElnOR = round(SElnOR,5)
    lnOR = round(lnOR,5)
    totalab = round(totalab,5)
    totalcd = round(totalcd,5)
    totalac = round(totalac,5)
    totalbd = round(totalbd,5)
    total = round(total,5)
    LowerlnOR = round(LowerlnOR,5)
    UpperlnOR = round(UpperlnOR,5)
    LowerOR = round(LowerOR,5)
    UpperOR = round(UpperOR,5)

    context = {'OA':OA,'OB':OB,'OR':OR,'SElnOR':SElnOR,'lnOR':lnOR,'a':a,'b':b,'c':c,'d':d,'totalab':totalab,'totalcd':totalcd,'totalac':totalac,
    'totalbd':totalbd,'total':total,'LowerlnOR':LowerlnOR,'UpperlnOR':UpperlnOR,'LowerOR':LowerOR,'UpperOR':UpperOR}
    return render(request, 'oddsRatio.html', context)

def bayesTheorem(request):
    A = get_request_param(request, 'A', 0.02739)
    B = get_request_param(request, 'B', 0.97261)
    AgivenB = get_request_param(request, 'AgivenB', 0.8)
    AgivenBwrong = get_request_param(request, 'AgivenBwrong', 0.2)

    PA1givenB = (A * AgivenB) / ((A * AgivenB) + (B * AgivenBwrong))
    PA1givenB = round(PA1givenB, 4)

    PA1givenB = round(PA1givenB,5)

    context = {'A':A,'B':B,'AgivenB':AgivenB, 'PA1givenB':PA1givenB,'AgivenBwrong':AgivenBwrong}
    return render(request, 'bayesTheorem.html', context)

def correlation(request):
    X = get_request_param(request, 'X', '575 571 570 575 304 320 340 344 340 350')
    Y = get_request_param(request, 'Y', '100 103 102 98 112 105 103 100 105 102')

    try:
        Xoutput = [int(x) for x in X.split()]
        Youtput = [int(x) for x in Y.split()]
    except ValueError:
        Xoutput = [float(x) for x in X.split()]
        Youtput = [float(x) for x in Y.split()]

    if len(Youtput) != len(Xoutput):
        lenY = int(len(Youtput))
        lenX = int(len(Xoutput))
        if lenY > lenX:
            for i in range(lenY-lenX):
                Xoutput.append(0)
        if lenX > lenY:
            for i in range(lenX-lenY):
                Youtput.append(0)

    def listToString(s):
        str1 = ""  
        for ele in s:  
            str1 += str(ele)
            str1 += ' '
        return str1

    Y = listToString(Youtput)
    X = listToString(Xoutput)

    productXY = []
    for num1, num2 in zip(Xoutput, Youtput):
        productXY.append(num1 * num2)
    sumXY = sum(productXY)

    Xmean = statistics.mean(Xoutput)
    Ymean = statistics.mean(Youtput)

    n = len(Xoutput)

    COV = (sumXY - n*Xmean*Ymean)/(n-1)

    VarianceX = statistics.variance(Xoutput)
    VarianceY = statistics.variance(Youtput)

    CorXY = COV/(VarianceX * VarianceY)**0.5

    pearsonR = pearsonr(Xoutput, Youtput)

    CorXY = round(CorXY,5)
    VarianceX = round(VarianceX,5)
    VarianceY = round(VarianceY,5)
    COV = round(COV,5)
    n = round(n,5)
    Xmean = round(Xmean,5)
    Ymean = round(Ymean,5)
    sumXY = round(sumXY,5)

    context = {'pearsonR':pearsonR,'Y':Y,'X':X,'CorXY':CorXY,'VarianceX':VarianceX,'VarianceY':VarianceY,'COV':COV,'n':n,'Xmean':Xmean,'Ymean':Ymean,'sumXY':sumXY}
    return render(request, 'correlation.html', context)

def Mannwhitneyu(request):
    Sample1 = get_request_param(request, 'Sample1', '1\n2\n3\n4')
    Sample2 = get_request_param(request, 'Sample2', '8\n5\n2\n3')

    try:
        Sample1Output = [int(x) for x in Sample1.split()]
        Sample2Output = [int(x) for x in Sample2.split()]
    except ValueError:
        Sample1Output = [float(x) for x in Sample1.split()]
        Sample2Output = [float(x) for x in Sample2.split()]

    MannWhitneyOutput = mannwhitneyu(Sample1Output, Sample2Output, use_continuity=True, alternative=None)

    pdmann = pd.DataFrame(MannWhitneyOutput)

    statistic = pdmann.iloc[0:1].values
    statistic = float(statistic[0])

    pvalue = pdmann.iloc[0:2].values
    pvalue = float(pvalue[1])

    statistic = round(statistic,5)
    pvalue = round(pvalue,5)

    context = {'statistic':statistic,'pvalue':pvalue,'MannWhitneyOutput':MannWhitneyOutput,'Sample1':Sample1,'Sample2':Sample2}

    return render(request, 'mannwhitneyu.html', context)

def Wilcoxon(request):
    Sample1 = get_request_param(request, 'Sample1', '21\n18\n16\n27\n20\n21\n20\n19\n25\n19\n26\n29\n20\n24\n16')
    Sample2 = get_request_param(request, 'Sample2', '17\n18\n18\n20\n25\n18\n16\n18\n20\n23\n24\n26\n23\n18\n15')

    try:
        Sample1Output = [int(x) for x in Sample1.split()]
        Sample2Output = [int(x) for x in Sample2.split()]
    except ValueError:
        Sample1Output = [float(x) for x in Sample1.split()]
        Sample2Output = [float(x) for x in Sample2.split()]

    Diff = []
    zip_object = zip(Sample1Output, Sample2Output)
    for list1_i, list2_i in zip_object:
        Diff.append(list1_i-list2_i)

    p = wilcoxon(Diff)
    pdwil = pd.DataFrame(p)
    statisticwil = pdwil.iloc[0:1].values
    statisticwil = float(statisticwil[0])
    pvaluewil = pdwil.iloc[0:2].values
    pvaluewil = float(pvaluewil[1])
    statisticwil = round(statisticwil,5)
    pvaluewil = round(pvaluewil,5)

    context = {'statistic':statisticwil,'pvalue':pvaluewil,'Sample1':Sample1,'Sample2':Sample2}

    return render(request, 'Wilcoxon.html', context)

def Kruskal(request):
    Sample1 = get_request_param(request, 'Sample1', '24\n30\n37\n39\n40\n45\n49\n70')
    Sample2 = get_request_param(request, 'Sample2', '32\n33\n36\n44\n44\n46\n58\n65')
    Sample3 = get_request_param(request, 'Sample3', '23\n30\n32\n37\n38\n40\n53\n65')
    Sample4 = get_request_param(request, 'Sample4', '27\n31\n36\n40\n42\n50\n62\n66')

    try:
        Sample1Output = [int(x) for x in Sample1.split()]
        Sample2Output = [int(x) for x in Sample2.split()]
        Sample3Output = [int(x) for x in Sample3.split()]
        Sample4Output = [int(x) for x in Sample4.split()]
    except ValueError:
        Sample1Output = [float(x) for x in Sample1.split()]
        Sample2Output = [float(x) for x in Sample2.split()]
        Sample3Output = [float(x) for x in Sample3.split()]
        Sample4Output = [float(x) for x in Sample4.split()]

    Kruskalnumber = stats.kruskal(Sample1Output, Sample2Output, Sample3Output, Sample4Output)

    pdmann = pd.DataFrame(Kruskalnumber)

    statistic = pdmann.iloc[0:1].values
    statistic= float(statistic[0])

    pvalue = pdmann.iloc[0:2].values
    pvalue = float(pvalue[1])
    statistic = round(statistic,5)
    pvalue = round(pvalue,5)

    context = {'statistic':statistic,'pvalue':pvalue,'Sample1':Sample1,'Sample2':Sample2,'Sample3':Sample3,'Sample4':Sample4}
    return render(request, 'Kruskal.html', context)

def Spearman(request):
    Sample1 = get_request_param(request, 'Sample1', '4\n3\n2\n1\n1\n3\n2\n1\n3\n4')
    Sample2 = get_request_param(request, 'Sample2', '5\n4\n3\n2\n1\n5\n4\n2\n2\n4')

    try:
        Sample1Output = [int(x) for x in Sample1.split()]
        Sample2Output = [int(x) for x in Sample2.split()]
    except ValueError:
        Sample1Output = [float(x) for x in Sample1.split()]
        Sample2Output = [float(x) for x in Sample2.split()]

    if len(Sample1Output) != len(Sample2Output):
        lenSample1 = int(len(Sample1Output))
        lenSample2 = int(len(Sample2Output))
        if lenSample1 > lenSample2:
            for i in range(lenSample1-lenSample2):
                Sample2Output.append(0)
        if lenSample2 > lenSample1:
            for i in range(lenSample2-lenSample1):
                Sample1Output.append(0)

    def listToString(s):
        str1 = ""  
        for ele in s:  
            str1 += str(ele)
            str1 += '\n'
        return str1

    Sample1 = listToString(Sample1Output)
    Sample2 = listToString(Sample2Output)

    corr = stats.spearmanr(Sample1Output, Sample2Output)

    pdmann = pd.DataFrame(corr)
    statistic = pdmann.iloc[0:1].values
    statistic = float(statistic[0])

    pvalue = pdmann.iloc[0:2].values
    pvalue = float(pvalue[1])
    statistic = round(statistic,5)
    pvalue = round(pvalue,5)

    context = {'statistic':statistic,'pvalue':pvalue,'Sample1':Sample1,'Sample2':Sample2}
    return render(request, 'Spearman.html', context)

def Variance(request):
    varianceList = get_request_param(request, 'varianceList', '1 2 3 4')

    STDlist = [int(x) for x in varianceList.split()]
    std = statistics.stdev(STDlist)
    variance = statistics.variance(STDlist)
    mean = statistics.mean(STDlist)
    n = len(STDlist)
    sumX = sum(STDlist)

    #Confidence interval
    UpperConfidence90 = mean + 1.645 * (variance/n)**0.5 #upper 90%
    LowerConfidence90 = mean - 1.645 * (variance/n)**0.5 #lower 90%

    UpperConfidence95 = mean + 1.96 * (variance/n)**0.5 #upper 95%
    LowerConfidence95 = mean - 1.96 * (variance/n)**0.5 #lower 95%

    UpperConfidence99 = mean + 2.576 * (variance/n)**0.5 #upper 99%
    LowerConfidence99 = mean - 2.576 * (variance/n)**0.5 #lower 99%

    mean = round(mean,5)
    std = round(std,5)
    variance = round(variance,5)
    n = round(n,5)
    UpperConfidence90 = round(UpperConfidence90,5)
    LowerConfidence90 = round(LowerConfidence90,5)
    LowerConfidence95 = round(LowerConfidence95,5)
    UpperConfidence95 = round(UpperConfidence95,5)
    LowerConfidence99 = round(LowerConfidence99,5)
    UpperConfidence99 = round(UpperConfidence99,5)

    context = {'mean':mean,'std':std,'variance':variance,'n':n,'UpperConfidence90':UpperConfidence90,
    'LowerConfidence90':LowerConfidence90,'LowerConfidence95':LowerConfidence95,
    'UpperConfidence95':UpperConfidence95,'LowerConfidence99':LowerConfidence99,'UpperConfidence99':UpperConfidence99,'varianceList':varianceList, 'sumX':sumX}
    return render(request, 'Variance.html', context)

def experimentalR(request):
    context = {}
    return render(request, 'experimentalR.html', context)

def aboutUs(request):
    return render(request, 'footerPages/aboutUs.html')

def privacyPolicy(request):
    return render(request, 'footerPages/privacyPolicy.html')

def termsofUse(request):
    return render(request, 'footerPages/termsofUse.html')

def contact(request):
    return render(request, 'footerPages/contact.html')