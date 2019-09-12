import numpy as np

S = input("Enter the string: ")

# Let p(c = 'a') = m. You are given the following prior for m:
# p(m = 0.1) = 0.9
# p(m = 0.3) = 0.04
# p(m = 0.5) = 0.03
# p(m = 0.7) = 0.02
# p(m = 0.9) = 0.01

cnt = 0   # observation counts
prior = [0.9, 0.04, 0.03, 0.02, 0.01]   # prior probabilities
m = [0.1, 0.3, 0.5, 0.7, 0.9]   # likelihood or m

for i in range(len(S)):
    if S[i] == 'a':
        cnt += 1

observed = cnt/len(S)   # observed probability  
posterior = np.multiply(m, prior)/observed   # posterior probabilities 


# Printing output
for i in range(len(posterior)):
    print("p(m = {:.1f} | data) = {:.4f}".format(m[i], posterior[i]))
i_max = np.argmax(posterior)
m_hat = (observed*posterior[i_max])/posterior[i_max]
print("p(c = 'a' | data) = {:.4f}".format(m_hat))

