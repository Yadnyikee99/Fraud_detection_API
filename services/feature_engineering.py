import pandas as pd
import numpy as np 

def create_features(transaction):
    df= pd.DataFrame([transaction])

    df['amount_to_balance_ratio'] = (
        df['amount'] / np.where(df['oldbalanceOrg']== 0,1, df['oldbalanceOrg']))
    df['is_zero_balance'] = (
        (df['oldbalanceOrg'] == 0).astype(int)
        )
                                                                                                    
    df['balancediff_Dest_including_amount'] = (
        (df['newbalanceDest']-df['amount']) - df['oldbalanceDest']
    )

    df['balancediff_Org_including_amount'] = (
        (df['newbalanceOrig']-df['amount']) - df['oldbalanceOrg']
    )

    df = pd.get_dummies(df,
                        columns=["type"], drop_first= True
                        )
    print(df.columns)
    return df