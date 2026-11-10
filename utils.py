import pandas as pd
import numpy as np

def create_samples(df_train , sample_size , data_overlap , seed):
    """
    Takes the training data df_train, requested sample_size per sample, and data_overlap (between 0 and 1) between the samples.

    Returns X_train_1 , y_train_1 , X_train_2 , y_train_2 , obv_MAD , obv_MAD_to_average . 
    (MAD - mean absolute difference, between the number of observations per month between the two samples) 

    """

    if not 0 <= data_overlap <= 1:
        raise ValueError("data_overlap must be between 0 and 1")

        
    df_train = df_train.copy()

    features = ['annual_inc' , 'loan_amnt' , 'int_rate' , 'installment']
    target = 'y'

    shared_obv_per_sample = round(data_overlap*sample_size)
    unique_obv_per_sample = sample_size - shared_obv_per_sample
    
    total_observations = 2*unique_obv_per_sample + shared_obv_per_sample

    if total_observations > len(df_train):
        raise ValueError(
            f"Need {total_observations} observations, "
            f"but df_train only has {len(df_train)}"
        )

    df_train = df_train.iloc[:total_observations]
    
    # shuffling, then splitting
    df_train_shuffled = df_train.sample(frac = 1 , random_state = seed).reset_index(drop = True)

    df_sample_1 = df_train_shuffled.iloc[:sample_size]
    df_sample_2 = df_train_shuffled.iloc[unique_obv_per_sample : unique_obv_per_sample + sample_size]

    X_train_1 = df_sample_1[features]
    y_train_1 = df_sample_1[target]

    X_train_2 = df_sample_2[features]
    y_train_2 = df_sample_2[target]

     # computing MAD - mean absolute difference, between the number of observations per month between the two samples
    df_obv_per_month_1 = df_sample_1.groupby('issue_d' , as_index = False).agg(month_obv_1 = ('loan_amnt' , 'size'))         
    df_obv_per_month_2 = df_sample_2.groupby('issue_d' , as_index = False).agg(month_obv_2 = ('loan_amnt' , 'size'))
    df_obv_per_month_both = pd.merge(df_obv_per_month_1 , df_obv_per_month_2 , on = 'issue_d' , how = 'outer').fillna(0)
    df_obv_per_month_both['obv_abs_difference'] = np.abs(df_obv_per_month_both['month_obv_1'] - df_obv_per_month_both['month_obv_2'])
    obv_MAD = df_obv_per_month_both['obv_abs_difference'].mean()
    
    mean_1 = df_obv_per_month_both["month_obv_1"].mean()
    mean_2 = df_obv_per_month_both["month_obv_2"].mean()
    obv_MAD_to_average = obv_MAD / min(mean_1, mean_2)    
    
    return X_train_1 , y_train_1 , X_train_2 , y_train_2 , obv_MAD , obv_MAD_to_average





def models_comparison_metrics(m1_name , m2_name , y_pred_1 , y_prob_1 , y_pred_2 , y_prob_2 , y_test , sample_size , data_overlap , exp_number , obv_MAD , obv_MAD_to_average , seed):
    """
    Compute similarity, agreement, and error metrics between two binary classifiers.

    Parameters
    ----------
    y_pred_1 : array-like
        Binary predictions from model 1.
    y_prob_1 : array-like
        Predicted probabilities for the positive class from model 1.
    y_pred_2 : array-like
        Binary predictions from model 2.
    y_prob_2 : array-like
        Predicted probabilities for the positive class from model 2.
    y_test : array-like
        True binary labels.

    Returns
    -------
    dict
        Dictionary containing prediction/probability correlations,
        joint prediction probabilities, agreement and Jaccard similarity,
        overall error correlation, false-positive/false-negative rates,
        and false-negative overlap metrics, including expected and empirical
        joint FN rates, FN lift, FN excess, FN correlation, and FN Jaccard.

    """

    pred_corr = np.corrcoef(y_pred_1 , y_pred_2)[0,1]
    prob_corr = np.corrcoef(y_prob_1 , y_prob_2)[0,1]

    P_1_1 = ((y_pred_1 == 1) & (y_pred_2 == 1)).mean()
    P_1_0 = ((y_pred_1 == 1) & (y_pred_2 == 0)).mean()
    P_0_1 = ((y_pred_1 == 0) & (y_pred_2 == 1)).mean()
    P_0_0 = ((y_pred_1 == 0) & (y_pred_2 == 0)).mean()
    agreement_rate = P_1_1 + P_0_0

    
    intersection = ((y_pred_1 == 1) & (y_pred_2 == 1)).sum()
    union = ((y_pred_1 == 1) | (y_pred_2 == 1)).sum()
    jaccard = intersection / union

    err_1 = (y_pred_1 != y_test).astype(int)
    err_2 = (y_pred_2 != y_test).astype(int)
    err_corr = np.corrcoef(err_1, err_2)[0, 1]

    
    m1_fp_rate = ((y_pred_1 == 1) & (y_test == 0)).sum() / (y_test == 0).sum()
    m2_fp_rate = ((y_pred_2 == 1) & (y_test == 0)).sum() / (y_test == 0).sum()

    
    FN_1 = (y_pred_1[y_test == 1] == 0).astype(int)
    FN_2 = (y_pred_2[y_test == 1] == 0).astype(int)

    m1_fn_rate = FN_1.mean()
    m2_fn_rate = FN_2.mean()
    
    expected_FN_both = m1_fn_rate * m2_fn_rate
    empirical_FN_both = (FN_1 & FN_2).mean()
    
    FN_lift = empirical_FN_both / expected_FN_both if expected_FN_both > 0 else np.nan
    FN_excess = empirical_FN_both - expected_FN_both
    
    FN_corr = np.corrcoef(FN_1, FN_2)[0, 1] if FN_1.std() > 0 and FN_2.std() > 0 else np.nan
    
    FN_union = (FN_1 | FN_2).sum()
    FN_jaccard = (FN_1 & FN_2).sum() / FN_union if FN_union > 0 else np.nan
    
    return {
        'exp_num' : exp_number ,
        'model1' : m1_name , 
        'model2' : m2_name ,
        'sample_size' : sample_size ,
        'data_overlap' : data_overlap ,
        'pred_corr': pred_corr,
        'prob_corr': prob_corr,
        'P_1_1': P_1_1,
        'P_1_0': P_1_0,
        'P_0_1': P_0_1,
        'P_0_0': P_0_0,
        'agreement_rate': agreement_rate,
        'jaccard': jaccard,
        'err_corr': err_corr,
        'm1_fn_rate': m1_fn_rate,
        'm1_fp_rate': m1_fp_rate,
        'm2_fn_rate': m2_fn_rate,
        'm2_fp_rate': m2_fp_rate ,
        'expected_FN_both': expected_FN_both,
        'empirical_FN_both': empirical_FN_both,
        'FN_lift': FN_lift,
        'FN_excess': FN_excess,
        'FN_corr': FN_corr,
        'FN_jaccard': FN_jaccard,
        'MAD' : obv_MAD ,
        'MAD_to_average' : obv_MAD_to_average ,
        'seed' : seed
        }
    


def train_test_models(m1 , m2 , X_train_1 , y_train_1 , X_train_2 , y_train_2 , X_test):
    """
    Train two classifiers on separate training samples and generate
    predictions on the same test set.

    Parameters
    ----------
    m1, m2 : sklearn-compatible classifiers
        Models supporting fit(), predict(), and predict_proba().
    X_train_1, X_train_2 : array-like
        Feature matrices used to train models 1 and 2.
    y_train_1, y_train_2 : array-like
        Target labels used to train models 1 and 2.
    X_test : array-like
        Feature matrix on which both models are evaluated.

    Returns
    -------
    y_pred_1, y_prob_1, y_pred_2, y_prob_2
        Class predictions and positive-class probabilities
        generated by both models.

    """
    model1 = m1.fit(X_train_1 , y_train_1)
    model2 = m2.fit(X_train_2 , y_train_2)
    
    y_pred_1 = model1.predict(X_test)
    y_prob_1 = model1.predict_proba(X_test)[:,1]

    y_pred_2 = model2.predict(X_test)
    y_prob_2 = model2.predict_proba(X_test)[:,1]

    
    return y_pred_1 , y_prob_1 , y_pred_2 , y_prob_2









    