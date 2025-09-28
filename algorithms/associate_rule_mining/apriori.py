import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules


class Apriori:

    def __init__(self) -> None:
        pass

    def convert_into_binary(self, x):
        # Return boolean directly instead of 0/1
        return x > 0

    def mba(self, df, sup, lif, conf):
        basket = pd.pivot_table(
            data=df,
            index='InvoiceNo',
            columns='Description',
            values='Quantity',
            aggfunc='sum',
            fill_value=0
        )

        # Apply binary conversion → already True/False
        basket_sets = basket.applymap(self.convert_into_binary)

        try:
            basket_sets.drop(columns=['POSTAGE'], inplace=True)
        except KeyError:
            pass

        frequent_itemsets = apriori(basket_sets, min_support=sup, use_colnames=True)
        rules_mlxtend = association_rules(frequent_itemsets, metric="lift", min_threshold=0)

        # Filter by lift and confidence
        rules_mlxtend = rules_mlxtend[
            (rules_mlxtend['lift'] >= lif) & (rules_mlxtend['confidence'] >= conf)
        ]
        return rules_mlxtend

    def get_associate_rules(self, ecommerce_df, customer_clusters_kmeans):
        df = pd.merge(ecommerce_df, customer_clusters_kmeans, on='CustomerID', how='inner')
        df['Description'] = df['Description'].str.strip()

        # Segment customers by cluster
        data0 = df[df.Cluster == 0]
        data1 = df[df.Cluster == 1]
        data2 = df[df.Cluster == 2]
        data3 = df[df.Cluster == 3]

        data = [
            self.mba(data0, 0.01, 4, 0.8),
            self.mba(data1, 0.01, 4, 0.8),
            self.mba(data2, 0.01, 4, 0.8),
            self.mba(data3, 0.01, 4, 0.8)
        ]

        result = pd.concat(data)
        return result
