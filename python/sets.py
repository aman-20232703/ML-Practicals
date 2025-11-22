# sets and its operations
s={1,2,4,5,"aman",12.67,11}
s1={4,5,6,7,11}
print(s,"\n")
print(len(s),"\n")
s.add(43)
print(s)
s.update([34,88])
print(s,"\n")
s.remove(2)
print(s,"\n")
s.discard(4)
print(s,"\n")

a=s.union(s1)
print(a,"\n")
b=s.intersection(s1)
print(b,"\n")
c=s.difference(s1)
print(c,"\n")
d=s.symmetric_difference(s1)
print(d)