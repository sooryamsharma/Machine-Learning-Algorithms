import numpy as np

p = np.random.uniform(low=0.0, high=1.0, size=10000)
S = ""
for prob in np.nditer(p):
    if prob <= 0.1:
        S = S+'a'
    else:
        S = S+'b'

cnt = 0
firstCnt = 0
secondCnt = 0
thirdCnt = 0
fourthCnt = 0
finalCnt = 0

for i in range(len(S)):
    if S[i] == 'a':
        cnt += 1
        prob = cnt/(i+1)
        if prob < 0.08:
            firstCnt += 1
            #print(prob)
        if prob >= 0.08 and prob < 0.09:
            secondCnt += 1
            #print(prob)
        if prob >= 0.09 and prob <= 0.11:
            thirdCnt += 1
            #print(prob)
        if prob > 0.11 and prob <= 0.12:
            fourthCnt += 1
            #print(prob)
        if prob > 0.12:
            finalCnt += 1
            #print(prob)

print("In {:d} of the simulations p(c = 'a') < 0.08.".format(firstCnt))
print("In {:d} of the simulations p(c = 'a') < 0.09.".format(firstCnt + secondCnt))
print("In {:d} of the simulations p(c = 'a') is in the interval [0.09, 0.11].".format(thirdCnt))
print("In {:d} of the simulations p(c = 'a') > 0.11.".format(fourthCnt + finalCnt))
print("In {:d} of the simulations p(c = 'a') > 0.12.".format(finalCnt))





