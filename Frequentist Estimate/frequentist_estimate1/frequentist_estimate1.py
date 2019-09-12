import numpy as np

p = np.random.uniform(low=0.0, high=1.0, size=3100)
S = ""
for prob in np.nditer(p):
    if prob <= 0.1:
        S = S+'a'
    else:
        S = S+'b'

cnt = 0
for c in S:
    if c == 'a':
        cnt += 1
        
#frequentist_prob = cnt/len(S)
print('p(c = \'a\') = {:.4f}'.format(cnt/len(S)))

