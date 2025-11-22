#dictionary and its operations
students={'name':'aman', 'age':21, 'roll':20232703}
print(students)
print(students.get('age'))
students['address']='delhi'
print(students)
students.update({'age':20})
print(students)
students.pop('address')
students
print(students)
print(students.keys())
print(students.values())
print(students.items())
print("name" in students)
print('address' not in students)
for keys in students:
    print(keys)