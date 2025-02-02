import numpy as np, pandas as pd

df = pd.DataFrame([
    ['Sunny', 'Warm', 'Normal', 'Strong', 'Warm', 'Same', 'yes'],
    ['Sunny', 'Warm', 'High', 'Strong', 'Warm', 'Same', 'yes'],
    ['Rainy', 'Cold', 'High', 'Strong', 'Warm', 'Change', 'no'],
    ['Sunny', 'Warm', 'High', 'Strong', 'Cool', 'Change', 'yes']
], columns=['Sky', 'Temp', 'Humidity', 'Wind', 'Water', 'Forecast', 'EnjoySport'])

concepts, target = df.iloc[:, :-1].values, df.iloc[:, -1].values

def learn(c, t):
    s, g = c[0].copy(), [['?'] * len(c[0]) for _ in range(len(c[0]))]
    for i, h in enumerate(c):
        if t[i] == "yes":
            for x in range(len(s)):
                if h[x] != s[x]: s[x], g[x][x] = '?', '?'
        else:
            for x in range(len(s)):
                g[x][x] = s[x] if h[x] != s[x] else '?'
    return s, [h for h in g if h != ['?'] * len(s)]

print("Final Specific Hypothesis:", (s := learn(concepts, target))[0])
print("Final General Hypothesis:", s[1])
