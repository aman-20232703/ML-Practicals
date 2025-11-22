# list and its operations
p=[1,2,3,4,5,6,"aman",12.67,True]
print("",p,"\n")
p.append(10)
print("",p,"\n")
p.remove(4)
print(p,"\n")
p.pop()
print("",p,"\n")
p[2]=13
print("",p,"\n")

q=[]
for i in range(1,9):
    q.append(i)
print("",q)

# print[list(map(int,input().split()))]

# slicing and indexing
print("",len(p))
print("",p[2:7])
print("",p[-5:-2])
