#!/usr/bin/env python
# coding: utf-8

# In[63]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dfmat = pd.read_csv('student-mat.csv')
dfpor = pd.read_csv('student-por.csv')

dfmat.head()


# In[64]:


import datetime 

Fecha = datetime.datetime.now()

print("La fecha en la que se ejecuta este script es:")
print(Fecha.strftime('%d/%m/%Y %H:%M:%S'))


# In[65]:


dfpor.head()


# In[66]:


dfmat.isna().sum()


# In[67]:


dfmat.dropna(axis = 1)


# In[68]:


dfpor.isna().sum()


# In[69]:


dfpor.dropna(axis = 1)


# In[70]:


School_GP=dfmat[dfmat['school']=='GP']
School_GP.shape


# In[71]:


School_MS=dfmat[dfmat['school']=='MS']
School_MS.shape


# In[72]:


fig, ax = plt.subplots()
School_MS.sex.value_counts().plot(kind = "pie", labels = ["Mujeres", "Hombres"], title = "Escuela MS Porcentaje de hombres y mujeres para el curso de matemáticas en la Escuela MS")
plt.show()


# In[73]:


fig, ax = plt.subplots()
School_GP.sex.value_counts().plot(kind = "pie", labels = ["Mujeres", "Hombres"], title = "Escuela MS Porcentaje de hombres y mujeres para el curso de matemáticas en la Escuela MS")
plt.show()


# In[74]:


School_GP.groupby("age").size().plot(kind = "bar", title = "Edades en el curso de matemáticas para Escuela GP")
plt.show()


# In[75]:


School_MS.groupby("age").size().plot(kind = "bar", title = "Edades en el curso de matemáticas para Escuela MS")
plt.show()


# In[76]:


print("Promedio de las edades en la escuela GP: ")
print(School_GP['age'].mean())


# In[77]:


print("Promedio de las edades en la escuela MS: ")
print(School_MS['age'].mean())


# In[78]:


print("El promedio para estudiantes los estudiantes de la escuela MS fueron los siguientes: ")
print("Promedio G1: ")
print(School_MS['G1'].mean())


# In[79]:


print("Promedio G2: ")
print(School_MS['G2'].mean())


# In[80]:


print("Promedio G3: ")
print(School_MS['G3'].mean())


# In[81]:


print("El promedio para estudiantes los estudiantes de la escuela GP fueron los siguientes: ")
print("Promedio G1: ")
print(School_GP['G1'].mean())


# In[82]:


print("Promedio G2: ")
print(School_GP['G2'].mean())


# In[83]:


print("Promedio G3: ")
print(School_GP['G3'].mean())


# In[84]:


School_MS.groupby("G1").size().plot(kind = "bar", title = "Promedio de notas G1 para la Escuela MS en el curso de matemáticas")
plt.show()
School_MS.groupby("G2").size().plot(kind = "bar", title = "Promedio de notas G2 para la Escuela MS en el curso de matemáticas")
plt.show()
School_MS.groupby("G3").size().plot(kind = "bar", title = "Promedio de notas G3 para la Escuela MS en el curso de matemáticas")
plt.show()


# In[85]:


School_GP.groupby("G1").size().plot(kind = "bar", title = "Promedio de notas G1 para la Escuela GP en el curso de matemáticas")
plt.show()
School_GP.groupby("G2").size().plot(kind = "bar", title = "Promedio de notas G2 para la Escuela GP en el curso de matemáticas")
plt.show()
School_GP.groupby("G3").size().plot(kind = "bar", title = "Promedio de notas G3 para la Escuela GP en el curso de matemáticas")
plt.show()


# In[87]:


Primera_Nota=[]
Segunda_Nota=[]
Tercera_Nota=[]

for i in School_MS['G1']:
    Primera_Nota.append(i)
    for h in School_MS['G2']:
        Segunda_Nota.append(h)
        for z in School_MS['G3']:
                Tercera_Nota.append(z)
    


# In[88]:


Notasfinales=[]
for o in range (0,len(School_MS.index)):
    notafinal=(Primera_Nota[o]+Segunda_Nota[o]+Tercera_Nota[o])/3
    Notasfinales.append(notafinal)
print(Notasfinales)


# In[111]:


Ausencias = []
Total = School_MS['absences'].max()
for j in School_MS['absences']:
    Ausencias.append(j)
print(Ausencias)
print("\n")
print(f"El total de ausencias es {Total}")


# In[108]:


asistencias=[]

for x in range (0,len(School_MS.index)):
    
    Ausenciasnum = ( - ( (Ausencias[x]/Total) * 100) + 100)    
    asistencias.append(Ausenciasnum)
    
print(asistencias)
print("\n")
print(len(asistencias))


# In[109]:


extr=[]
approvd=[]
for R in range(0,len(Notasfinales)):
    Asist = asistencias[R]
    Grades = Notasfinales[R]
    
    if(Asist < 80):
        approvd.append(0)
        extr.append(None)
        
    elif (Asist > 80 and Grades < 10):
        approvd.append(0)
        extr.append(None)
        
    elif(Asist >80 and Grades >= 10 and Grades <= 15):
        approvd.append(1)
        extr.append(1)
        
    elif(Asist > 80 and Grades > 15):
        approvd.append(1)
        extr.append(0)
        
print(extr)
print(approvd)


# In[113]:


School_MS.insert(33,'approved',approvd,allow_duplicates = False)
School_MS.insert(34,'extra',extr,allow_duplicates = False)


# In[114]:


School_MS


# In[116]:


num = 1
for k in asistencias:
    print(f'Para el estudiante {num} de la Escuela MS en el curso de matemáticas su porcentaje de asistencia fue de un {(round(k,2))}%')
    num+=1


# In[118]:


Primera_Nota2=[]
Segunda_Nota2=[]
Tercera_Nota2=[]

for i in School_GP['G1']:
    Primera_Nota2.append(i)
    for h in School_GP['G2']:
        Segunda_Nota2.append(h)
        for z in School_GP['G3']:
                Tercera_Nota2.append(z)


# In[120]:


Notasfinales2=[]
for h in range (0,len(School_GP.index)):
    notafinal2=(Primera_Nota2[h]+Segunda_Nota2[h]+Tercera_Nota2[h])/3
    Notasfinales2.append(notafinal2)
print(Notasfinales2)


# In[121]:


Ausencias2 = []
Total2 = School_GP['absences'].max()
for j in School_GP['absences']:
    Ausencias2.append(j)
print(Ausencias2)
print("\n")
print(f"El total de ausencias es {Total2}")


# In[122]:


asistencias2=[]

for x in range (0,len(School_GP.index)):
    
    Ausenciasnum2 = ( - ( (Ausencias2[x]/Total2) * 100) + 100)    
    asistencias2.append(Ausenciasnum2)
    
print(asistencias2)
print("\n")
print(len(asistencias2))


# In[139]:


extr2=[]
approvd2=[]
for R in range(0,len(Notasfinales2)):
    Asist2 = asistencias2[R]
    Grades2 = Notasfinales2[R]
    
    if(Asist2 < 80):
        approvd2.append(0)
        extr2.append(None)
        
    elif (Asist2 > 80 and Grades2 < 10):
        approvd2.append(0)
        extr2.append(None)
        
    elif(Asist2 > 80 and Grades2 >= 10 and Grades2 <= 15):
        approvd2.append(1)
        extr2.append(1)
        
    elif(Asist2 > 80 and Grades2 > 15):
        approvd2.append(1)
        extr2.append(0)
        
print(extr2)
print("\n")
print(approvd2)


# In[138]:


School_GP.insert(33,'approved',approvd2,allow_duplicates = False)
School_GP.insert(34,'extra',extr2,allow_duplicates = False)


# In[136]:


School_GP


# In[140]:


num2 = 1
for k in asistencias:
    print(f'Para el estudiante {num2} de la Escuela GP en el curso de matemáticas su porcentaje de asistencia fue de un {(round(k,2))}%')
    num2+=1


# In[141]:


School_GP=dfpor[dfpor['school']=='GP']
School_GP.shape


# In[148]:


School_MS=dfpor[dfpor['school']=='MS']
School_MS.shape


# In[149]:


# In[72]:


fig, ax = plt.subplots()
School_MS.sex.value_counts().plot(kind = "pie", labels = ["Mujeres", "Hombres"], title = "Escuela MS Porcentaje de hombres y mujeres para el curso de portugues en la Escuela MS")
plt.show()


# In[73]:


fig, ax = plt.subplots()
School_GP.sex.value_counts().plot(kind = "pie", labels = ["Mujeres", "Hombres"], title = "Escuela MS Porcentaje de hombres y mujeres para el curso de portugues en la Escuela MS")
plt.show()


# In[74]:


School_GP.groupby("age").size().plot(kind = "bar", title = "Edades en el curso de portugues para Escuela GP")
plt.show()


# In[75]:


School_MS.groupby("age").size().plot(kind = "bar", title = "Edades en el curso de portugues para Escuela MS")
plt.show()


# In[76]:


print("Promedio de las edades en la escuela GP: ")
print(School_GP['age'].mean())


# In[77]:


print("Promedio de las edades en la escuela MS: ")
print(School_MS['age'].mean())


# In[78]:


print("El promedio para estudiantes los estudiantes de la escuela MS fueron los siguientes: ")
print("Promedio G1: ")
print(School_MS['G1'].mean())


# In[79]:


print("Promedio G2: ")
print(School_MS['G2'].mean())


# In[80]:


print("Promedio G3: ")
print(School_MS['G3'].mean())


# In[81]:


print("El promedio para estudiantes los estudiantes de la escuela GP fueron los siguientes: ")
print("Promedio G1: ")
print(School_GP['G1'].mean())


# In[82]:


print("Promedio G2: ")
print(School_GP['G2'].mean())


# In[83]:


print("Promedio G3: ")
print(School_GP['G3'].mean())


# In[84]:


School_MS.groupby("G1").size().plot(kind = "bar", title = "Promedio de notas G1 para la Escuela MS en el curso de portugues")
plt.show()
School_MS.groupby("G2").size().plot(kind = "bar", title = "Promedio de notas G2 para la Escuela MS en el curso de portugues")
plt.show()
School_MS.groupby("G3").size().plot(kind = "bar", title = "Promedio de notas G3 para la Escuela MS en el curso de portugues")
plt.show()


# In[85]:


School_GP.groupby("G1").size().plot(kind = "bar", title = "Promedio de notas G1 para la Escuela GP en el curso de portugues")
plt.show()
School_GP.groupby("G2").size().plot(kind = "bar", title = "Promedio de notas G2 para la Escuela GP en el curso de portugues")
plt.show()
School_GP.groupby("G3").size().plot(kind = "bar", title = "Promedio de notas G3 para la Escuela GP en el curso de portugues")
plt.show()


# In[87]:


Primera_Nota=[]
Segunda_Nota=[]
Tercera_Nota=[]

for i in School_MS['G1']:
    Primera_Nota.append(i)
    for h in School_MS['G2']:
        Segunda_Nota.append(h)
        for z in School_MS['G3']:
                Tercera_Nota.append(z)
    


# In[88]:


Notasfinales=[]
for o in range (0,len(School_MS.index)):
    notafinal=(Primera_Nota[o]+Segunda_Nota[o]+Tercera_Nota[o])/3
    Notasfinales.append(notafinal)
print(Notasfinales)


# In[111]:


Ausencias = []
Total = School_MS['absences'].max()
for j in School_MS['absences']:
    Ausencias.append(j)
print(Ausencias)
print("\n")
print(f"El total de ausencias es {Total}")


# In[108]:


asistencias=[]

for x in range (0,len(School_MS.index)):
    
    Ausenciasnum = ( - ( (Ausencias[x]/Total) * 100) + 100)    
    asistencias.append(Ausenciasnum)
    
print(asistencias)
print("\n")
print(len(asistencias))


# In[109]:


extr=[]
approvd=[]
for R in range(0,len(Notasfinales)):
    Asist = asistencias[R]
    Grades = Notasfinales[R]
    
    if(Asist < 80):
        approvd.append(0)
        extr.append(None)
        
    elif (Asist > 80 and Grades < 10):
        approvd.append(0)
        extr.append(None)
        
    elif(Asist >80 and Grades >= 10 and Grades <= 15):
        approvd.append(1)
        extr.append(1)
        
    elif(Asist > 80 and Grades > 15):
        approvd.append(1)
        extr.append(0)
        
print(extr)
print(approvd)


# In[113]:


School_MS.insert(33,'approved',approvd,allow_duplicates = False)
School_MS.insert(34,'extra',extr,allow_duplicates = False)


# In[114]:


School_MS


# In[116]:


num = 1
for k in asistencias:
    print(f'Para el estudiante {num} de la Escuela MS en el curso de matemáticas su porcentaje de asistencia fue de un {(round(k,2))}%')
    num+=1


# In[118]:


Primera_Nota2=[]
Segunda_Nota2=[]
Tercera_Nota2=[]

for i in School_GP['G1']:
    Primera_Nota2.append(i)
    for h in School_GP['G2']:
        Segunda_Nota2.append(h)
        for z in School_GP['G3']:
                Tercera_Nota2.append(z)


# In[120]:


Notasfinales2=[]
for h in range (0,len(School_GP.index)):
    notafinal2=(Primera_Nota2[h]+Segunda_Nota2[h]+Tercera_Nota2[h])/3
    Notasfinales2.append(notafinal2)
print(Notasfinales2)


# In[121]:


Ausencias2 = []
Total2 = School_GP['absences'].max()
for j in School_GP['absences']:
    Ausencias2.append(j)
print(Ausencias2)
print("\n")
print(f"El total de ausencias es {Total2}")


# In[122]:


asistencias2=[]

for x in range (0,len(School_GP.index)):
    
    Ausenciasnum2 = ( - ( (Ausencias2[x]/Total2) * 100) + 100)    
    asistencias2.append(Ausenciasnum2)
    
print(asistencias2)
print("\n")
print(len(asistencias2))


# In[139]:


extr2=[]
approvd2=[]
for R in range(0,len(Notasfinales2)):
    Asist2 = asistencias2[R]
    Grades2 = Notasfinales2[R]
    
    if(Asist2 < 80):
        approvd2.append(0)
        extr2.append(None)
        
    elif (Asist2 > 80 and Grades2 < 10):
        approvd2.append(0)
        extr2.append(None)
        
    elif(Asist2 > 80 and Grades2 >= 10 and Grades2 <= 15):
        approvd2.append(1)
        extr2.append(1)
        
    elif(Asist2 > 80 and Grades2 > 15):
        approvd2.append(1)
        extr2.append(0)
        
print(extr2)
print("\n")
print(approvd2)


# In[138]:


School_GP.insert(33,'approved',approvd2,allow_duplicates = False)
School_GP.insert(34,'extra',extr2,allow_duplicates = False)


# In[150]:


# In[136]:


School_GP


# In[151]:


# In[140]:


num2 = 1
for k in asistencias:
    print(f'Para el estudiante {num2} de la Escuela GP en el curso de matemáticas su porcentaje de asistencia fue de un {(round(k,2))}%')
    num2+=1


# In[ ]:




