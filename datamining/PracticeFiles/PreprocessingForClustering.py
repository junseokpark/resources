import pandas as pd
import numpy as np


def paretoXY(s, x, y,threshold=0):
    ''' Return a list until you account for X% of the whole and remainders are less than Y% individually.
        The threshold percentage allows you to hedge your bets and check items just past the treshold. '''
    #Generate percentages, and sort, and find accumulated total
    #Exclude any negative payments that can make the cumulative percentage go > 100% before we get to them
    df=pd.DataFrame( s[s>0].sort_values(ascending=False) )
    df['pc']= 100*s/s.sum()
    df['cum_pc']=df['pc'].cumsum()
    #Return ordered items that either together account at least for X% of the total,
    # and/or individually account for at least Y% of the total
    #The threshold provides a fudge factor on both components...
    return df[ (df['cum_pc']-df['pc'] <= x+ x*threshold/100) | (df['pc'] >= y-y*threshold/100) ]


# reference from http://www.kimberlycoffey.com/blog/2016/8/k-means-clustering-for-customer-segmentation

# a customer-level dataset
# [columns]
# frequency : the number of invoices with purchases during the year
# recency : refers to the number of days that have elapsed since the customer last purchased something
#           (so, smaller numbers indicate more recent activity on the customer’s account)
# monetary value data : the amount that the customer spent during the year (minus monetary data rest to zero)
def createCustomerLevelDataSet(df):

    result_df = pd.DataFrame({'CustomerID':df['CustomerID'].unique().astype(int)})
    df["recency"] = df["InvoiceDate"].apply(lambda x: pd.Timestamp('2011-12-10') - x)

    # remove returns so only consider the data of most recent *purchase*
    temp_df = df[df.purchaseinvoice == 1]

    # Obtain # of days since most recent purchase
    recency_df = temp_df.loc[temp_df.groupby('CustomerID')['recency'].idxmin()]

    # Add recency to customer data
    result_df = result_df.set_index('CustomerID').join(recency_df.set_index('CustomerID')[['recency']])
    result_df['recency'] = result_df['recency'].dt.days
    result_df = result_df[result_df['recency'] > 0]

    # Frequency
    invoices_df = df[['CustomerID', 'InvoiceNo', 'purchaseinvoice']].drop_duplicates().sort_values(by=['CustomerID'])
    invoices_df = invoices_df.groupby(['CustomerID'])['purchaseinvoice'].agg('sum').to_frame()
    invoices_df = invoices_df.rename(columns={'purchaseinvoice': 'frequency'})
    invoices_df = invoices_df.reset_index()
    result_df = result_df.join(invoices_df.set_index("CustomerID"))
    result_df = result_df[result_df.frequency > 0]

    # Monetary value of customers
    df['amount'] = df['Quantity'] * df['UnitPrice']
    annualsale_df = df.groupby(['CustomerID'])['amount'].agg('sum').to_frame()
    annualsale_df = annualsale_df.reset_index()

    # Add monetary value to customer data
    result_df = result_df.join(annualsale_df.set_index("CustomerID"))
    result_df = result_df.rename(columns={'amount': 'monetary'})
    result_df['monetary'] = np.where(result_df.monetary < 0, 0, result_df.monetary)


    # 20-80 Rule (The Pareto Principle)
    # 20% represent the high-value, important customers a business would want to protect
    result_df = result_df.sort_values(by='monetary', ascending=False)
    pareto_cutoff = result_df['monetary'].sum() * 0.8
    result_df['pareto'] = np.where(result_df['monetary'].cumsum() <= pareto_cutoff, "Top 20%", "Bottom 80%")
    result_df['pareto'] = pd.Series(result_df['pareto'], dtype='category')
    pd.crosstab(index=result_df['pareto'], columns="count")


    # Preprocess data
    # standardized input variables
    # transform our three input variables to reduce positive skew and then standardize them as z-scores.
    cols = list(result_df.columns)
    cols.remove('pareto')
    #cols.remove('CustomerID')
    for col in cols:
        col_zscore = col + '_zscore'
        result_df[col_zscore] = (result_df[col] - result_df[col].mean()) / result_df[col].std(ddof=0)


    result_df['recency_log'] = np.log(result_df['recency'])
    result_df['frequency_log'] = np.log(result_df['frequency'])
    result_df['monetary_log'] = np.log(result_df['monetary'] + 0.1)


    return result_df


if __name__ == '__main__':
    df = pd.read_excel("data/OnlineRetail.xlsx")

    # dataframe information
    print(df.head())
    print(df.shape)

    # Remove any observations with missing ID numbers
    print("Current rows="+str(df.shape[0]))
    df_subset = df.dropna()
    print("Current rows after removing N/A=" + str(df_subset.shape[0]))

    # Restrict the date rang to one full year
    df_subset = df_subset[df_subset.InvoiceDate >= '2010-12-09']
    print("Current rows after restrict year=" + str(df_subset.shape[0]))

    # Restrict country to reduce variation by geography
    df_subset = df_subset[df_subset.Country == "United Kingdom"]
    print("Current rows after restrict country=" + str(df_subset.shape[0]))

    # Identify Returns
    df_subset["itemreturn"] = np.where(df_subset['InvoiceNo'].str.contains("C", na=False), True, False)
    df_subset["purchaseinvoice"] = np.where(df_subset["itemreturn"]==True,0,1)
    print(df_subset.head())

    df_customer = createCustomerLevelDataSet(df_subset)
    df_customer.to_csv("data/OnlineRetailPreprocessed.tab", sep = "\t")

