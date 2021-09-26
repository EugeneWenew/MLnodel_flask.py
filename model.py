import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def pred_values(input_data):
    df = pd.read_excel('d:\set.xlsx', header=0)
    df_1 = df.drop(['Персона'], axis=1)
    X = df_1
    StandardScaler().fit_transform(X)
    kmeans = KMeans(init="k-means++", n_clusters=4, n_init=12)
    kmeans.fit(X)
    X_test = pd.DataFrame(input_data)
    X_test = X_test.transpose()
    scaled_test = StandardScaler().fit_transform(X_test)
    predicted_label = kmeans.predict(scaled_test)
    return predicted_label

df_test = [24, 6, 348, 139419]
print(pred_values(df_test))







