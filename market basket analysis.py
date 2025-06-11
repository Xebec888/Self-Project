import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from functools import reduce

# These settings ensure that the full DataFrame is shown
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.max_rows', None)     # Show all rows
pd.set_option('display.max_colwidth', None) # Show full content in each cell
pd.set_option('display.expand_frame_repr', False)  # Prevent breaking across lines

#transaction dataset
transactions = [
    ['curd', 'sour cream'], ['curd', 'orange', 'sour cream'], ['bread', 'cheese', 'butter'], ['bread', 'butter'], ['bread', 'milk'], ['apple', 'orange', 'pear'],
    ['bread', 'milk', 'eggs'], ['tea', 'lemon'], ['curd', 'sour cream', 'apple'], ['eggs', 'wheat flour', 'milk'], ['pasta', 'cheese'], ['bread', 'cheese'],
    ['pasta', 'olive oil', 'cheese'], ['curd', 'jam'], ['bread', 'cheese', 'butter'], ['bread', 'sour cream', 'butter'], ['strawberry', 'sour cream'],
    ['curd', 'sour cream'], ['bread', 'coffee'], ['onion', 'garlic']
]

encoder = TransactionEncoder()
encoded_array = encoder.fit(transactions).transform(transactions) #to run the data through mlxtend apriori algorithm, need to transform transaction data into a one-hot encoded boolean array
df_itemsets = pd.DataFrame(encoded_array, columns= encoder.columns_) #Convert from one-hot encoded array to pandas dataframe and choosing column as encoder columns

print(' Number of transactions: ', len(transactions))
print('Number of unique items: ', len(set(sum(transactions, []))))
# To print the total number of items in transaction

frequent_itemsets = apriori(df_itemsets, min_support= 0.1 , use_colnames=True) #identifying frequent itemsets
print(frequent_itemsets)

#modifying dataset to only show 2 or more itemsets support metrics
frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda itemset: len(itemset))
print(frequent_itemsets[frequent_itemsets['length'] >= 2])

rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5) #generating association rules
print(rules.iloc[:, 0:7])

#visualizing Association rules
#using lambda function to convert the values of the antecedents and consequents columns from rules dataframe into strings
rules_plot = pd.DataFrame()
rules_plot['antecedents']= rules['antecedents'].apply(lambda x: ','.join(list(x)))
rules_plot['consequents']= rules['consequents'].apply(lambda x: ','.join(list(x)))
rules_plot['lift']= rules['lift'].apply(lambda x: round(x, 2))

pivot = rules_plot.pivot(index = 'antecedents', columns = 'consequents' , values = 'lift') #transform newly created rules_plot dataframe into matrix
print(pivot)

#Extract components into separate variables
antecedents = list(pivot.index.values)
consequents = list(pivot.columns)
pivot = pivot.to_numpy()

#visualizing using heatmap from matplotlib
fig, ax = plt.subplots()
im = ax.imshow(pivot, cmap = 'Reds') #converts the data from the pivot array into a color coded 2d image and cmap is using reds mapping
ax.set_xticks(np.arange(len(consequents)))
ax.set_yticks(np.arange(len(antecedents)))
ax.set_xticklabels(consequents)
ax.set_yticklabels(antecedents)
plt.setp(ax.get_xticklabels(), rotation=45 , ha="right", rotation_mode="anchor") #setp() method used to rotate the x-axis lables

for i in range(len(antecedents)):
    for j in range(len(consequents)):
        if not np.isnan(pivot[i, j]):
            text = ax.text(j, i, pivot[i, j], ha="center", va="center") #filter out any pair with no lift value

ax.set_title("Lift metric for frequent itemsets")
fig.tight_layout()
plt.show()

#Generating recommendations
butter_antecedent = rules[rules['antecedents'] == {'butter'}][['consequents', 'confidence']].sort_values('confidence', ascending = False)
#here we sort the rules by the confidence column so that the rules with the highest confidence rating appear at the beginning of the butter_antecedent dataframe

butter_consequents = [list(item) for item in butter_antecedent.iloc[0:3:,]['consequents']]
#in this list comprehension looping over the cosequents column in the butter_antecedent dataframe , picking up the first three values based on the butter_consequents list

item = 'butter'
print('Items frequently bought together with', item, 'are:', butter_consequents)

#planning discounts based on association rules
rules['itemsets'] = rules[['antecedents', 'consequents']].apply(lambda x: reduce(frozenset.union, x), axis = 1)
print(rules[['antecedents', 'consequents', 'itemsets']])

rules.drop_duplicates(subset=['itemsets'], keep='first', inplace=True)
print(rules['itemsets'])


discounted = []
others = []
for itemset in rules['itemsets']:
    for i, item in enumerate(itemset):
        if item not in others:
            discounted.append(item)
            itemset = set(itemset)
            itemset.discard(item)
            others.extend(itemset)
            break
        if i == len(itemset) - 1:
            discounted.append(item)
            itemset = set(itemset)
            itemset.discard(item)
            others.extend(itemset)
print(discounted)

print(list(set(discounted)))