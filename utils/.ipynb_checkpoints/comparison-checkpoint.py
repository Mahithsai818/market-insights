import sys
sys.dont_write_bytecode = True

import pandas as pd
from algorithms.associate_rule_mining import apriori


def get_best_clusters(ecommerce_df, customer_clusters_kmeans, customer_clusters_dbscan, customer_clusters_hierarchial):

    apriori_model = apriori.Apriori()
    
    associate_rules_kmeans = apriori_model.get_associate_rules(ecommerce_df, customer_clusters_kmeans)
    associate_rules_dbscan = apriori_model.get_associate_rules(ecommerce_df, customer_clusters_dbscan)
    associate_rules_hierarchial = apriori_model.get_associate_rules(ecommerce_df, customer_clusters_hierarchial)

    # Rename lift columns
    associate_rules_kmeans.rename(columns={"lift": "lift1"}, inplace=True)
    associate_rules_kmeans.drop(['leverage', 'conviction','antecedent support','consequent support','support','confidence'], axis=1, inplace=True)

    associate_rules_dbscan.rename(columns={"lift": "lift2"}, inplace=True)
    associate_rules_dbscan.drop(['leverage', 'conviction','antecedent support','consequent support', 'support','confidence'], axis=1, inplace=True)

    associate_rules_hierarchial.rename(columns={"lift": "lift3"}, inplace=True)
    associate_rules_hierarchial.drop(['leverage', 'conviction','antecedent support','consequent support','support','confidence'], axis=1, inplace=True)

    # Merge all clusters
    df = pd.merge(
        pd.merge(associate_rules_kmeans, associate_rules_dbscan, on=['antecedents','consequents'], how="inner"),
        associate_rules_hierarchial,
        on=['antecedents','consequents'],
        how="inner"
    )

    kmeans = dbscan = hierar = 0
    for _, row in df.iterrows():
        l1, l2, l3 = row['lift1'], row['lift2'], row['lift3']
        if l1 > l2 and l1 > l3:
            kmeans += 1
        elif l2 > l1 and l2 > l3:
            dbscan += 1
        elif l3 > l1 and l3 > l2:
            hierar += 1

    # Return best cluster
    if kmeans == max(kmeans, dbscan, hierar):
        print("best cluster by kmeans")
        return customer_clusters_kmeans
    elif dbscan == max(kmeans, dbscan, hierar):
        print("best cluster by dbscan")
        return customer_clusters_dbscan
    else:
        print("best cluster by hierarchial")
        return customer_clusters_hierarchial


def get_best_associate_rules(associate_rules_apriori, associate_rules_fpgrowth):

    # Compute Zhang metric for Apriori
    associate_rules_apriori['zhang metric1'] = None
    for i, row in associate_rules_apriori.iterrows():
        supp_x, supp_y, supp_xy = row['antecedent support'], row['consequent support'], row['support']
        max_supp = max(supp_x, supp_y)
        zhang_metric = (supp_xy - supp_x * supp_y) / (max_supp - supp_x * supp_y)
        associate_rules_apriori.at[i, 'zhang metric1'] = zhang_metric

    associate_rules_apriori.drop(['leverage', 'conviction','antecedent support','consequent support','support','confidence','lift'], axis=1, inplace=True)

    # Compute Zhang metric for FP-Growth
    associate_rules_fpgrowth['zhang metric2'] = None
    for i, row in associate_rules_fpgrowth.iterrows():
        supp_x, supp_y, supp_xy = row['antecedent support'], row['consequent support'], row['support']
        max_supp = max(supp_x, supp_y)
        zhang_metric = (supp_xy - supp_x * supp_y) / (max_supp - supp_x * supp_y)
        associate_rules_fpgrowth.at[i, 'zhang metric2'] = zhang_metric

    associate_rules_fpgrowth.drop(['leverage', 'conviction','antecedent support','consequent support','support','confidence','lift'], axis=1, inplace=True)

    # Merge rules
    df = pd.merge(
        associate_rules_apriori,
        associate_rules_fpgrowth,
        on=['antecedents','consequents'],
        how="inner"
    )

    # Count which rules are better
    ap = fp = 0
    for _, row in df.iterrows():
        if row['zhang metric1'] > row['zhang metric2']:
            ap += 1
        else:
            fp += 1

    # Return the better set
    if ap >= fp:
        print("best rules by apriori")
        return associate_rules_apriori
    else:
        print("best rules by fpgrowth")
        return associate_rules_fpgrowth
