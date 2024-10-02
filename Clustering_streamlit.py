import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

st.markdown("<h1 style='text-align: center;'>Customer Segmentation</h1>", unsafe_allow_html=True)

df = pd.read_csv ('Mall_Customers.csv')

df.rename(index=str, columns={
    'Annual Income (k$)' : 'Income',
    'Spending Score (1-100)' : 'Score'
}, inplace=True)

X = df.drop(['CustomerID', 'Gender'], axis=1)

st.header("Isi Dataset")
st.write(df)

#menampilkan elbow
cluster = []
for i in range (1,10):
    km = KMeans(n_clusters=i).fit(X)
    cluster.append(km.inertia_)

fig, ax = plt.subplots(figsize=(12,8))
sns.lineplot(x=list(range(1,10)), y =cluster, ax=ax)
ax.set_title('Mencari Elbow')
ax.set_xlabel('Clusters')
ax.set_ylabel('Inertia')

#tanda panah elbow
ax.annotate('Possible elbow point', xy=(3, 140000), xytext=(3, 50000), xycoords='data', 
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3', color='blue',lw=2))

ax.annotate('Possible elbow point', xy=(5, 80000), xytext=(5, 150000), xycoords='data', 
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3', color='blue',lw=2))

st.header("Elbow")
elbow_plot = st.pyplot(fig)

st.sidebar.header("Nilai Jumlah K (klaster)")
clust = st.sidebar.slider("Piih jumlah cluster :", 2,10,3,1)

def k_means(n_clust):
    kmeans = KMeans (n_clusters=n_clust).fit(X)
    X['Labels'] = kmeans.labels_

    plt.figure(figsize=(10, 8))
    sns.scatterplot(x='Income', y='Score', hue='Labels', data=X, 
                    palette=sns.color_palette('hls', n_clust), s=100)

    for label in sorted(X['Labels'].unique()):
        plt.annotate(label,
            (X[X['Labels'] == label]['Income'].mean(),
            X[X['Labels'] == label]['Score'].mean()),
            horizontalalignment='center',
            verticalalignment='center',
            size=12, weight='bold', color='black')
    
    st.header("Cluster Plot")
    st.pyplot(plt)
    st.write(X)

k_means(clust)



