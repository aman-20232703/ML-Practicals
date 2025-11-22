# 2D_Array
import numpy as np
a=np.array([[1,2,3,4],[6,7,8,9]])
print(a)

# all its operations
import numpy as np
a=np.array([[1,2,3,4],[6,7,8,9]])
b=np.array([[5,7,3,8]])
print(a.shape)
print(a.dtype)
print(a.ndim)
print(type(a))
print(np.concatenate((a,b)))
print(a+b)
print(np.multiply(a,b))
print(np.square(b))
print(a[1,2])
print(a[0:2, 0:3])


# array statistical values
import numpy as np
marks=np.array([45,56,87,69,75])
print(np.average(marks))
print(min(marks), max(marks))


import numpy as np
number=np.random.randint(1,100, size=10)
print(number)
print(np.mean(number))
print(np.std(number))
print(np.var(number))


# class work
import numpy as np
students_marks=np.array([[15,43,63],  #hindi
                   [52,36,89],  #english
                   [45,63,23]   #math
                  ]) 
print(np.average(students_marks))
print(np.max(students_marks))

import numpy as np
students_subjects=np.array(['hindi','english','math'])
students_marks=np.array([[15,43,63],  #hindi
                   [52,36,89],  #english
                   [45,63,23]   #math
                  ]) 
for students_subject, students_mark in zip(students_subjects,students_marks):
    print(f"{students_subject}:{students_mark}")

print(np.average(students_mark))
print(np.max(students_mark))