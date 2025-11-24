This project uses gene expression data from GSE10245 to separate two lung cancer subtypes: adenocarcinoma (AD) and squamous cell carcinoma (SCC). There are 58 samples total (40 AD, 18 SCC).

Project steps:

1) Load the data using GEOparse.
2) Run k-means clustering with k=2 to group the samples.
3) Create a DataFrame with each sample’s ID, its cluster number, and its true subtype (from the GEO metadata).
4) Calculate accuracy, which is the percentage of samples that were placed in the wrong group.
5) Split the data in half (balanced by subtype), train the model on one half, label the clusters, and then predict the subtype for the second half.
6) Compute accuracy for the training data and the testing data.
7) Make a bar plot showing three accuracy values:
   - accuracy using all data
   - accuracy using the training half
   - accuracy on the testing half

This project shows how an unsupervised method like k-means can be compared to real biological labels.
