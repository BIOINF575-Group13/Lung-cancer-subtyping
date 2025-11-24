from sklearn.preprocessing import StandardScaler


#Example of a data frame
'''Sample_ID	Subtype	Subtype_Int
0	GSM258551	adenocarcinoma	0
1	GSM258552	adenocarcinoma	0
2	GSM258553	squamous cell carcinoma	1
3	GSM258554	adenocarcinoma	0
4	GSM258555	squamous cell carcinoma	1
5	GSM258556	squamous cell carcinoma	1
6	GSM258557	squamous cell carcinoma	1
7	GSM258558	adenocarcinoma	0
8	GSM258559	adenocarcinoma	0
'''

#2. Perform clustering - you can use the sklearn.cluster module for this task.
ad_samples = metadata[metadata['Subtype'] == 'adenocarcinoma']
scc_samples = metadata[metadata['Subtype'] == 'squamous cell carcinoma']

# Take first 20 AD and first 9 SCC for train
train_df = pd.concat([ad_samples.head(20), scc_samples.head(9)])

# Take the remaining 20 AD and 9 SCC for test
test_df = pd.concat([ad_samples.tail(20), scc_samples.tail(9)])

# Features to use for clustering
feature_cols = ['Cluster']  # replace with your numeric columns

# Separate features
X_train = train_df[feature_cols]
X_test = test_df[feature_cols]
X_full = metadata[feature_cols]

# StandardScaler
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_full_scaled = scaler.transform(X_full)
