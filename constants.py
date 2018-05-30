from string import ascii_lowercase
index2alpha = list(ascii_lowercase)
index2alpha.extend([ "'", "-", "START_TOKEN", "END_TOKEN" ])

alpha2index = {}
for i,e in enumerate(index2alpha):
    alpha2index[e] = i

