import numpy as np 


X = np.array([4,3])

#Norm: Chuẩn hóa để tính ra một đại lượng biểu diễn độ lớn của một vector

#L0Norm: Số lượng các phần tử khác 0
#l0Norm=2
l0norm = np.linalg.norm(X, ord=0)
print(l0norm)

#L1Norm: Khoảng cách Mahattan
l1norm = np.linalg.norm(X, ord=1)
print(l1norm)

#L2Norm: Khoảng cách Euclid:  X=(a,b) => sqrt(a**2+b**2)
l2norm = np.linalg.norm(X, ord=2)
print(l2norm)









