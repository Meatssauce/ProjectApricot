from joblib import load, dump

with open('data/friendship-matrix.joblib', 'rb') as f:
    matrix = load(f)

# Verify the integrity of friendship_matrix
for name1 in matrix:
    for name2 in matrix[name1]:
        if matrix[name1][name2] != matrix[name2][name1]:
            raise ValueError(f'{name1}|{name2}')
print('No error detected')
