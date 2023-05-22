"""
This file uses the output of TestingPerformanceAnalysis.py, which is the file "pickles/comb_test_perf_case2.pkl", and
then selects 3 best combinations and plot those results.
It used the file "pickles/comb_test_perf_case2.pkl" as input.
"""
from MLPairsTrading import *


if __name__ == "__main__":
    print('start')

    # comb_list = ['isomap+agglomerative',
    #              'mds+agglomerative',
    #              't_sne+dbscan']

    comb_list = ['pca+agglomerative',
                 'isomap+agglomerative',
                 'mds+agglomerative']

    with open('pickles/comb_test_perf_case2.pkl', 'rb') as handle:
        comb_cum_ret = pickle.load(handle)

    best_perf_df = comb_cum_ret.filter(comb_list+['S&P500'])
    best_perf_df.plot()
    plt.legend(fontsize=10, loc='upper left')
    plt.title("Best combination strategies")
    plt.xlabel("Time")
    plt.ylabel("Cumulative Returns")
    plt.show()

    print('end')
