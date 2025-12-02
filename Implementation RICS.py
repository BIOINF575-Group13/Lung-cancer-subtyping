#We tried to identify the top 10 genes that are upregulated on the AC and SCC subtypes.

#To identified them, we need to translate the probe_ID to genes.
#Then, we need to perform log2 Fold Change to find the most upregulated and wilcoxon rank-sum test to identified statistically significant genes.


#To avoid adding the need to install new packages, log2Fold Change and wilcoxon rank-sum test will be performed using numpy and scipy.

#Afterwards, we took the top 10 statistically significant genes for each subtypes and z-score normalized their values across genes to visualize their changes in a cluster heatmap.


#################


# Importing necessary libraries
# Geoparse: Module to automatically download GEO datasets and parse them into pandas DataFrames for easy manipulation.
# Pandas: For data manipulation and analysis, providing data structures like DataFrames. Dependency for GEOparse.
# NumPy: For numerical operations to use alongside with pandas DataFrames.
# Sklearn: For machine learning tasks including data preprocessing, clustering algorithms, and evaluation metrics. 
# Importing only the required components from sklearn: StandardScaler for feature scaling, KMeans for clustering, and accuracy_score for evaluating model performance.
# Matplotlib: For data visualization to create plots and graphs.
# Scipy: For obtaining the statistical to calculate p-values with wilcoxon test
# Seanborn: To create the cluster heatmap
import GEOparse as geo
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns


# For faster processing, a custom function was developed to obtain the gene matrix and the cancer subtype of each sampel

def load_project_data(geo_accession: str, *args, **kwargs) -> list[pd.DataFrame]:
    """"
    Obtain the required data from GEO to use in this project.

    Args:
        geo_accession (str): The GEO accession number.
        destdir (str): The directory to save the downloaded GEO dataset.

    Returns:
        list: A list containing:
            expression_data (pd.DataFrame): The gene expression data.
            metadata (pd.DataFrame): The prepared metadata or the full metadata dataframe from the GEO dataset.
    """

    # Load GEO dataset
    gse = geo.get_GEO(geo=geo_accession, *args, **kwargs)

    # Extract expression data
    # Transposing to have genes as columns and samples as rows
    expression_data = gse.pivot_samples('VALUE').T

    metadata = gse.phenotype_data

    # Extracting the main column
    labels = metadata['title']
        
    # All the Cancer subtypes are between 2 "_"s
    # Splitting the strings by the underscore "_"
    # Keeping elements in second group (cancer subtype)
    # Making the pandas series to pandas dataframe
    labels = labels.str.split("_").str[1].to_frame()
        
    # Changing the column name in dataframe
    labels.rename(columns={'title': 'Subtype'}, inplace=True)

    # KMeans needs integers to compare the results
    # Making an integer representation of each cancer type
    # Making a dictionary with cancer type as key and number as value
    class2int = {cls: i for i, cls in enumerate(np.unique(labels))}
        
    # Creating a numpy vectorize for mapping the value based on the dictionary
    mapper = np.vectorize(class2int.get) 
        
    # Using mapper to create the integer representation of each value in the labels DF
    true_int = mapper(labels.to_numpy()) 
        
    # Adding the values to the dataframe
    labels['integer_label'] = true_int

    return [expression_data, labels]

# Re-running the data loading function to get fresh dataframes
matrix, labels = load_project_data(geo_accession = "GSE10245", 
                                     silent=True) # Adding silent argument to reduced output message

# Obtaining platform information to translate probe IDs to gene symbols
gpl = geo.get_GEO(geo="GPL570")
dataframe = gpl.table.copy() # copying dataframe to avoid SettingWithCopyWarning
# Keeping only relevant columns
dataframe = dataframe[['ID', 'Gene Symbol']]
# Renaming the ID column to match with expression matrix
dataframe.rename(columns={'ID': 'ID_REF'}, inplace=True)

# Reorganizing the expression matrix to have probe IDs as column for merging
matrix_T = matrix.T.reset_index()

# Merging both datafreames to have gene symbols
merged_df = pd.merge(matrix_T, dataframe, on='ID_REF', how='left')

# Cleaning the 'Gene Symbol' column to keep only the first gene symbol in case of multiple symbols
merged_df['Gene Symbol'] = merged_df['Gene Symbol'].str.split(" /// ").str[0]

# Removing rows with missing gene symbols
merged_df.dropna(subset=['Gene Symbol'], inplace=True)
# Dropping probe ID column
merged_df.drop(columns=['ID_REF'], inplace=True)
# Some rows have duplicated gene symbols, we will average their expression values
merged_df = merged_df.groupby('Gene Symbol').mean()
# Obtaining samples for each cancer subtype
ac_samples = labels[labels['Subtype'] == 'AC'].reset_index()['index']
scc_samples = labels[labels['Subtype'] == 'SCC'].reset_index()['index']
# Creating subset databframes for each cancer subtype
ac_df = merged_df[ac_samples]
scc_df = merged_df[scc_samples]


# Creating a results dataframe with log2 fold change and p-values
results_df = pd.DataFrame({
    # Getting the log2 fold change between the two cancer subtypes for each gene
    'log2_FC': np.log2(ac_df.mean(axis=1) / scc_df.mean(axis=1)),
    # Getting the p-values using Wilcoxon rank-sum test
    'p_value': stats.ranksums(ac_df, scc_df, axis=1).pvalue
})

# filtering significant genes with p-value < 0.05 and absolute log2 fold change > 1
significant_genes = results_df[(results_df['p_value'] < 0.05) & (results_df['log2_FC'].abs() > 1)].copy() # copying dataframe to avoid SettingWithCopyWarning

# Sorting by log2fold change in a descending order
significant_genes.sort_values(by='log2_FC', inplace=True,  ascending=False)

#taking top 10 upregulated and downregulated genes and keeping only the gene symbols
upregulated_genes = significant_genes.head(10).reset_index()['Gene Symbol']
downregulated_genes = significant_genes.tail(10).reset_index()['Gene Symbol']

# combining all the genes symbols in a single list
genes = pd.concat([upregulated_genes, downregulated_genes]).tolist()

# Keeping only the significant genes in the merged dataframe
heatmap_data = merged_df.loc[genes]
# Using the StandardScaler to normalize the data
scaler = StandardScaler()
# z-score normalization across genes
# genes are rows and StandardScaler works across columns, Transposing to do z-sccore and then returning to original format 
heatmap_data = pd.DataFrame(scaler.fit_transform(heatmap_data.T).T,
                            index=heatmap_data.index, 
                            columns=heatmap_data.columns)

# Combining the heatmap with the labels dataframe to obtain subtype information
# Splitting the steps in different lines for better readability, but keeping pandorable chaining style
heatmap_data = (pd
                .merge(heatmap_data.T, labels, left_index=True, right_index=True, how='inner', validate ="1:1") # To merge the dataframes, we need to transpose the heatmap data
                .set_index('Subtype') # adding subtype as index
                .drop(columns=['integer_label']) # dropping integer label column
                .T # Transposing back to original format
)

# Visualizing the results on a heatmap
plt.figure(figsize=(18, 15))
sns.clustermap(heatmap_data, 
               method='average', 
               cmap='vlag', 
               z_score=None, 
               col_cluster=True, 
               row_cluster=True, 
               yticklabels = True, 
               xticklabels = True, 
               linewidth=0.01)
plt.xlabel('Samples')   
plt.ylabel('Genes')
plt.show()


###### Interpretation

# This data shows that the SCC cancer subtype has a more homogenous expression of genes, in contrast to the AC subtype which has more variability in their gene expression
# In addition, we identified one AC and 3 SCC samples that are clustering with the other subtype. We confirmed the GSM manually to the information on GEOBUS to ensure there was no labeling error
# Samples were labeled AC and SCC based on histopathological findings. This same finding was reported by the authors of the paper. However, the paper does not address how these information could aid 
# the proper classification of all tumor. some of the markers that are present in most SCC are several Keratin proteins. Immunohistochemistry of several proteins has been used frequently in 
# skins tumors [https://onlinelibrary.wiley.com/doi/10.1111/j.1600-0625.2009.01006.x]. 
# This provide a clinically available method to ensure classification in cases in which there is doubt on the histopathological diagnosis or if the diagnosis is uncertain. 