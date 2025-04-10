import sklearn
import pandas

import numpy as np

# columns to use, all but the duplicate one
col_list=[]
for i in range(79):
    if i!=55:
        col_list.append(i)

data_headers="Destination Port, Flow Duration, Total Fwd Packets, Total Backward Packets,Total Length of Fwd Packets, Total Length of Bwd Packets, Fwd Packet Length Max, Fwd Packet Length Min, Fwd Packet Length Mean, Fwd Packet Length Std,Bwd Packet Length Max, Bwd Packet Length Min, Bwd Packet Length Mean, Bwd Packet Length Std,Flow Bytes/s, Flow Packets/s, Flow IAT Mean, Flow IAT Std, Flow IAT Max, Flow IAT Min,Fwd IAT Total, Fwd IAT Mean, Fwd IAT Std, Fwd IAT Max, Fwd IAT Min,Bwd IAT Total, Bwd IAT Mean, Bwd IAT Std, Bwd IAT Max, Bwd IAT Min,Fwd PSH Flags, Bwd PSH Flags, Fwd URG Flags, Bwd URG Flags, Fwd Header Length, Bwd Header Length,Fwd Packets/s, Bwd Packets/s, Min Packet Length, Max Packet Length, Packet Length Mean, Packet Length Std, Packet Length Variance,FIN Flag Count, SYN Flag Count, RST Flag Count, PSH Flag Count, ACK Flag Count, URG Flag Count, CWE Flag Count, ECE Flag Count, Down/Up Ratio, Average Packet Size, Avg Fwd Segment Size, Avg Bwd Segment Size, Fwd Header Length2,Fwd Avg Bytes/Bulk, Fwd Avg Packets/Bulk, Fwd Avg Bulk Rate, Bwd Avg Bytes/Bulk, Bwd Avg Packets/Bulk,Bwd Avg Bulk Rate,Subflow Fwd Packets, Subflow Fwd Bytes, Subflow Bwd Packets, Subflow Bwd Bytes,Init_Win_bytes_forward, Init_Win_bytes_backward, act_data_pkt_fwd, min_seg_size_forward,Active Mean, Active Std, Active Max, Active Min,Idle Mean, Idle Std, Idle Max, Idle Min, Label"
data_headers=data_headers.split(",")
#threat_classes="BENIGN, DoS Slowhttptest, PortScan, DDoS, Dos Hulk, FTP-Patator, Bot"
#threat_classes=threat_classes.split(",")
data_headers_count=len(data_headers)
# strip extra spaces from begin and end of headers
for i in range (data_headers_count):
    entry=data_headers[i]
    data_headers[i]=entry.lstrip(' ').strip(' ')


print(data_headers)
# Open the file in read mode
with open("data/output.01.csv","r") as infile:
    data=pandas.read_csv(infile,names=data_headers,usecols=col_list)
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.dropna(inplace=True)
    print(data.shape)

labels=np.asarray(data["Label"])

label_encoder = sklearn.preprocessing.LabelEncoder()
label_encoder.fit(labels)

labels = label_encoder.transform(labels)

print(labels)
print(list(label_encoder.classes_))




data_features = data.to_dict(orient='records')

vec = sklearn.feature_extraction.DictVectorizer()
features = vec.fit_transform(data_features).toarray()



features_train, features_test, labels_train, labels_test = sklearn.model_selection.train_test_split(
features, labels,
        test_size=0.20)








classifier = sklearn.ensemble.RandomForestClassifier()

from sklearn.experimental import enable_iterative_imputer



classifier.fit(features_train, labels_train)

accuracy_test = classifier.score(features_test, labels_test)

print ("Test Accuracy:", accuracy_test)

accuracy_train = classifier.score(features_train, labels_train)

print ("Train Accuracy:", accuracy_train)
