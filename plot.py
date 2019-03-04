import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

from os import listdir
from os.path import isfile, join

def save_plot_tsp(d,cols):
    plt.figure()
    for c in cols:
        if('SA' in c): plt.plot(d[c+'_fitness'], label=c)
    plt.title('SA')
    plt.legend(loc='lower right')
    plt.xlabel("Epochs x10")
    plt.ylabel("Fitness")
    plt.savefig("./graphs/SA_HP_tsp.png")

    plt.figure()
    for c in cols:
        if('GA' in c): plt.plot(d[c+'_fitness'], label=c)
    plt.title('GA')
    plt.legend(loc='lower right')
    plt.xlabel("Epochs x10")
    plt.ylabel("Fitness")
    plt.savefig("./graphs/GA_HP_tsp.png")

    plt.figure()
    for c in cols:
        if('MIMIC' in c): plt.plot(d[c+'_fitness'], label=c)
    plt.title('MIMIC')
    plt.legend(loc='lower right')
    plt.xlabel("Epochs x10")
    plt.ylabel("Fitness")
    plt.savefig("./graphs/MIMIC_HP_tsp.png")

    plt.figure()
    for c in ['rhc','SA0.75','GA100_50_10','MIMIC100_50_0.1']:
        plt.plot(d[c+'_fitness'], label=c)
    plt.title('Comparison on TSP')
    plt.legend(loc='lower right')
    plt.xlabel("Epochs x10")
    plt.ylabel("Fitness")
    plt.savefig("./graphs/Comparison_tsp.png")

def save_plot_ff(d,cols):
    plt.figure()
    for c in cols:
        if('SA' in c): plt.plot(d[c+'_fitness'], label=c)
    plt.title('SA')
    plt.legend(loc='lower right')
    plt.xlabel("Epochs x10")
    plt.ylabel("Fitness")
    plt.savefig("./graphs/SA_HP_ff.png")

    plt.figure()
    for c in cols:
        if('GA' in c): plt.plot(d[c+'_fitness'], label=c)
    plt.title('GA')
    plt.legend(loc='lower right')
    plt.xlabel("Epochs x10")
    plt.ylabel("Fitness")
    plt.savefig("./graphs/GA_HP_ff.png")

    plt.figure()
    for c in cols:
        if('MIMIC' in c): plt.plot(d[c+'_fitness'], label=c)
    plt.title('MIMIC')
    plt.legend(loc='lower right')
    plt.xlabel("Epochs x10")
    plt.ylabel("Fitness")
    plt.savefig("./graphs/MIMIC_HP_ff.png")

    plt.figure()
    for c in ['rhc','FLOP_SA0.15','FLOP_GA100_30_30','FLOP_MIMIC100_50_0.9']:
        plt.plot(d[c+'_fitness'], label=c)
    plt.title('Comparison on ff')
    plt.legend(loc='lower right')
    plt.xlabel("Epochs x10")
    plt.ylabel("Fitness")
    plt.savefig("./graphs/Comparison_ff.png")

def save_plot_cp(d,cols):
    plt.figure()
    for c in cols:
        if('SA' in c): plt.plot(d[c+'_fitness'], label=c)
    plt.title('SA')
    plt.legend(loc='lower right')
    plt.xlabel("Epochs x10")
    plt.ylabel("Fitness")
    plt.savefig("./graphs/SA_HP_cp.png")

    plt.figure()
    for c in cols:
        if('GA' in c): plt.plot(d[c+'_fitness'], label=c)
    plt.title('GA')
    plt.legend(loc='lower right')
    plt.xlabel("Epochs x10")
    plt.ylabel("Fitness")
    plt.savefig("./graphs/GA_HP_cp.png")

    plt.figure()
    for c in cols:
        if('MIMIC' in c): plt.plot(d[c+'_fitness'], label=c)
    plt.title('MIMIC')
    plt.legend(loc='lower right')
    plt.xlabel("Epochs x10")
    plt.ylabel("Fitness")
    plt.savefig("./graphs/MIMIC_HP_cp.png")

    plt.figure()
    for c in ['rhc','PEAKS_SA0.55','PEAKS_GA100_10_30','PEAKS_MIMIC100_50_0.9']:
        plt.plot(d[c+'_fitness'], label=c)
    plt.title('Comparison on CP')
    plt.legend(loc='lower right')
    plt.xlabel("Epochs x10")
    plt.ylabel("Fitness")
    plt.savefig("./graphs/Comparison_cp.png")

def save_plot(d,name):
    plt.figure()
    plt.plot(d['acc_trg'], label="Training")
    plt.plot(d['acc_val'], label="Validation")
    plt.plot(d['acc_tst'], label="Testing")
    plt.legend(loc='lower right')
    plt.title(name)
    plt.xlabel("Epochs x10")
    plt.ylabel("Accuracy")
    plt.savefig("./graphs/"+str(name)+".png")

def NN_graphs():
    mypath = './NN_OUTPUT/'
    data = pd.read_csv(mypath+'RHC_LOG.txt', sep=",")
    save_plot(data,"NN_rhc")

    files_sa = [f for f in listdir(mypath) if isfile(join(mypath, f)) and 'SA' in f]
    for f in files_sa:
        df = pd.read_csv(mypath+f, sep=",")
        save_plot(df,"NN_"+str(f[:-8]))

    files_ga = [f for f in listdir(mypath) if isfile(join(mypath, f)) and 'GA' in f]
    for f in files_ga:
        df = pd.read_csv(mypath+f, sep=",")
        save_plot(df,"NN_"+str(f[:-8]))

def TSP_graphs():
    mypath = './TSP_LOGS/'
    df = pd.read_csv(mypath+'TSP_RHC_1_LOG.txt', sep=",")
    df.rename(columns = {'fitness':'rhc_fitness'}, inplace = True)
    df.rename(columns = {'time':'rhc_time'}, inplace = True)

    cols = ['rhc_fitness']

    files_sa = [f for f in listdir(mypath) if isfile(join(mypath, f)) and 'SA' in f]
    for f in files_sa:
        d = pd.read_csv(mypath+f, sep=",")
        df = pd.merge(df,d,on='iterations')
        df.rename(columns = {'fitness':str(f[4:-10])+'_fitness'}, inplace = True)
        df.rename(columns = {'time':str(f[4:-10])+'_time'}, inplace = True)
        cols.append(str(f[4:-10]))

    files_ga = [f for f in listdir(mypath) if isfile(join(mypath, f)) and 'GA' in f]
    for f in files_ga:
        dg = pd.read_csv(mypath+f, sep=",")
        df = pd.merge(df,dg,on='iterations')
        df.rename(columns = {'fitness':str(f[4:-10])+'_fitness'}, inplace = True)
        df.rename(columns = {'time':str(f[4:-10])+'_time'}, inplace = True)
        cols.append(str(f[4:-10]))

    files_mimic = [f for f in listdir(mypath) if isfile(join(mypath, f)) and 'MIMIC' in f]
    for f in files_mimic:
        dg = pd.read_csv(mypath+f, sep=",")
        df = pd.merge(df,dg,on='iterations')
        df.rename(columns = {'fitness':str(f[4:-10])+'_fitness'}, inplace = True)
        df.rename(columns = {'time':str(f[4:-10])+'_time'}, inplace = True)
        cols.append(str(f[4:-10]))

    save_plot_tsp(df,cols)

def FF_graphs():
    mypath = './FLIPFLOP/'
    df = pd.read_csv(mypath+'FLIPFLOP_RHC_1_LOG.txt', sep=",")
    df.rename(columns = {'fitness':'rhc_fitness'}, inplace = True)
    df.rename(columns = {'time':'rhc_time'}, inplace = True)

    cols = ['rhc_fitness']

    files_sa = [f for f in listdir(mypath) if isfile(join(mypath, f)) and 'SA' in f]
    for f in files_sa:
        d = pd.read_csv(mypath+f, sep=",")
        df = pd.merge(df,d,on='iterations')
        df.rename(columns = {'fitness':str(f[4:-10])+'_fitness'}, inplace = True)
        df.rename(columns = {'time':str(f[4:-10])+'_time'}, inplace = True)
        cols.append(str(f[4:-10]))

    files_ga = [f for f in listdir(mypath) if isfile(join(mypath, f)) and 'GA' in f]
    for f in files_ga:
        dg = pd.read_csv(mypath+f, sep=",")
        df = pd.merge(df,dg,on='iterations')
        df.rename(columns = {'fitness':str(f[4:-10])+'_fitness'}, inplace = True)
        df.rename(columns = {'time':str(f[4:-10])+'_time'}, inplace = True)
        cols.append(str(f[4:-10]))

    files_mimic = [f for f in listdir(mypath) if isfile(join(mypath, f)) and 'MIMIC' in f]
    for f in files_mimic:
        dg = pd.read_csv(mypath+f, sep=",")
        df = pd.merge(df,dg,on='iterations')
        df.rename(columns = {'fitness':str(f[4:-10])+'_fitness'}, inplace = True)
        df.rename(columns = {'time':str(f[4:-10])+'_time'}, inplace = True)
        cols.append(str(f[4:-10]))

    save_plot_ff(df,cols)

def CP_graphs():
    mypath = './CONTPEAKS/'
    df = pd.read_csv(mypath+'CONTPEAKS_RHC_1_LOG.txt', sep=",")
    df.rename(columns = {'fitness':'rhc_fitness'}, inplace = True)
    df.rename(columns = {'time':'rhc_time'}, inplace = True)

    cols = ['rhc_fitness']

    files_sa = [f for f in listdir(mypath) if isfile(join(mypath, f)) and 'SA' in f]
    for f in files_sa:
        d = pd.read_csv(mypath+f, sep=",")
        df = pd.merge(df,d,on='iterations')
        df.rename(columns = {'fitness':str(f[4:-10])+'_fitness'}, inplace = True)
        df.rename(columns = {'time':str(f[4:-10])+'_time'}, inplace = True)
        cols.append(str(f[4:-10]))

    files_ga = [f for f in listdir(mypath) if isfile(join(mypath, f)) and 'GA' in f]
    for f in files_ga:
        dg = pd.read_csv(mypath+f, sep=",")
        df = pd.merge(df,dg,on='iterations')
        df.rename(columns = {'fitness':str(f[4:-10])+'_fitness'}, inplace = True)
        df.rename(columns = {'time':str(f[4:-10])+'_time'}, inplace = True)
        cols.append(str(f[4:-10]))

    files_mimic = [f for f in listdir(mypath) if isfile(join(mypath, f)) and 'MIMIC' in f]
    for f in files_mimic:
        dg = pd.read_csv(mypath+f, sep=",")
        df = pd.merge(df,dg,on='iterations')
        df.rename(columns = {'fitness':str(f[4:-10])+'_fitness'}, inplace = True)
        df.rename(columns = {'time':str(f[4:-10])+'_time'}, inplace = True)
        cols.append(str(f[4:-10]))

    save_plot_cp(df,cols)

if __name__ == '__main__':
    NN_graphs()
    TSP_graphs()
    FF_graphs()
    CP_graphs()
    #iteration,MSE_trg,MSE_val,MSE_tst,acc_trg,acc_val,acc_tst,elapsed
    #iterations,fitness,time
