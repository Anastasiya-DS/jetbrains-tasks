import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def generate_data(N=10000):
    t = np.arange(N)
    
    cpu = 40 + 5 * np.sin(t / 50) + np.random.normal(0, 2, N)
    mem = 50 + 2 * np.cos(t / 100) + np.random.normal(0, 1, N)
    lat = 100 + np.random.normal(0, 5, N)
    err = np.random.poisson(0.1, N)
    
    incident = np.zeros(N)  
    incident_starts = np.random.choice(range(200, N - 100), 25, replace=False)
    
    for start in incident_starts:
        duration = 20
        lead_up = 40      
        cpu[start - lead_up : start] += np.linspace(0, 25, lead_up)
        lat[start - lead_up : start] += np.linspace(0, 80, lead_up)    
        cpu[start : start + duration] += 50
        lat[start : start + duration] += 150
        err[start : start + duration] += 12
        incident[start : start + duration] = 1
        
    return pd.DataFrame({
        "cpu": cpu, "mem": mem, "latency": lat, "errors": err, "incident": incident
    })

def sliding_windows(data, W=30, H=20):
    feature_cols = ['cpu', 'mem', 'latency', 'errors']
    raw_values = data[feature_cols].values
    target_values = data['incident'].values
    
    X = []
    y = []
    
    for t in range(W, len(data) - H):
        window = raw_values[t - W : t]
        X.append(window.flatten())
        
        future_period = target_values[t : t + H]
        if np.any(future_period == 1):
            y.append(1)
        else:
            y.append(0)
            
    return np.array(X), np.array(y)