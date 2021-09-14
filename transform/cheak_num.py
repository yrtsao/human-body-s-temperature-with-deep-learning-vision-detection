tph = 40   
tpl = 30
rgh = 255
rgl = 0
mean = 137
tpp = 35
x = ((mean-tpl)*(tph-tpl)/(rgh-rgl)+tpl)
print(x)
y = (((tpp-tpl)*(rgh-rgl)+tpl)/(tph-tpl))+tpl
print(y)


import heapq
nums = [1, 8, 2, 23, 7, -4, 18, 23, 42, 37, 2, 100, 100]
print(heapq.nlargest(3, nums)) # Prints [42, 37, 23]
# print(heapq.nsmallest(3, nums)) # Prints [-4, 1, 2]


