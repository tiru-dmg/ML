import csv

def find_s(data):
    h = ['0'] * (len(data[0]) - 1)
    for ex in data:
        if ex[-1] == 'Yes':
            for i in range(len(h)):
                h[i] = ex[i] if h[i] == '0' else '?' if h[i] != ex[i] else h[i]
    return h

data = [
    ['Sunny', 'Warm', 'Normal', 'Strong', 'Warm', 'Same', 'Yes'],
    ['Sunny', 'Warm', 'High', 'Strong', 'Warm', 'Same', 'Yes'],
    ['Rainy', 'Cold', 'High', 'Strong', 'Warm', 'Change', 'No'],
    ['Sunny', 'Warm', 'High', 'Strong', 'Cool', 'Change', 'Yes']
]

print("Most Specific Hypothesis:", find_s(data))
