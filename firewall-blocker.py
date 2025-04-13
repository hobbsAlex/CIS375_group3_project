import pandas as pd 
import requests
from requests.auth import HTTPBasicAuth


api_url = 'https://192.168.10.1/api'
api_key = 'x1nGwK8V0a6DD9TGOOVouq2ShY2eUAEDd7oKU0eeMCogqEwXbwQ4uGIZq8c9or3kFKV/u+HeTYZ+fq0H'
api_secret = '8N7Nbj9G+fKo60NLLPhdCJ4JHE6qzwDLalfelNDspdrOIqSiJozKk7oGMm09x1zLcX3u8hQA1Gzi1Jos'
alias_name = 'BlockedIPs'


file_path = "output.01.csv"

df = pd.read_csv(file_path)

source_ip_col = df.columns[-1]
label_col = df.columns[-2]


non_benign_ips = df[df[label_col] != "BENIGN"][source_ip_col].unique()

def add_ip_to_alias(ip):
    url = f"{api_url}/firewall/alias_util/add/{alias_name}"
    payload = {
        "address": ip,
        "description": "Blocked by AI model",
    }

    response = requests.post(url, json=payload, auth=HTTPBasicAuth(api_key, api_secret), verify=False)


    if response.status_code == 200:
        print(f"[+] Blocked IP: {ip}")
    else: 
        print(f"[!] Failed to block {ip}: {response.status_code}, {response.text}")


for ip in non_benign_ips:
    add_ip_to_alias(ip)


