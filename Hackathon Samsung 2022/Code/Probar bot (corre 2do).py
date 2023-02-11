#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tkinter
from tkinter import *

#Importamos las librerias previamente instaladas. 

import nltk
from nltk.stem.lancaster import LancasterStemmer #Este es un modulo que se encuentra en la libreria NLTK que nos proporciona las herramientas necesarias para el lenguaje natural
import tensorflow
import tflearn

#Agregamos dos librerias adicionales para responderle al usuario.
import random #Sirve para responder aleatoriamene cuando ya conoces la categoria a la que corresponde la frase ingresada por el usuario. 
import numpy as np
import pickle #Libreria que sirve para guardar temporales de manera permanente.

stemmer=LancasterStemmer()


#Aqui empieza la parte VISUAL

base = Tk()
base.title("Chatbot")
base.geometry("400x500")

#Esta funcion se encarga de convertir la frase que el usuario ingresa en 1s y 0s para luego poderla 
#ingresar al modelo de prediccion.
def bag_of_words(s, words):
    #la variable S contiene la informacion que el usuario ingresa
    #En la variable words contengo la bolsa de palabras 
    
    bag = [0 for _ in range(len(words))]

    #"Hola como estas, puedo reservar una cita?"
    s_words = nltk.word_tokenize(s) #Convierto la frase ingresada por el usuario en palabras 
    #["Hola", "como", "estas", "puedo", "reservar", "una", "cita"]
    
    s_words = [stemmer.stem(word.lower()) for word in s_words]
    #["Hola", "como", "estar", "poder", "reservar", "una", "cita"]
    
        #Que esta contenido en la variable words?
        #["'s", 'acceiv', 'anyon', 'ar', 'bye', 'card', 'cash', 'credit', 'day', 'de', 'do',
        #'form', 'good', 'goodby', 'hello', 'help', 'hi', 'hour', 'how', 'is', 'lat', 'mastercard', 
        #'nuev', 'on', 'op', 'salud', 'see', 'tak', 'thank', 'that', 'ther', 'today', 'up', 'what', 
        #'when', 'yo', 'you']
    
    
    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
            
    return np.array(bag)


def chatbot_response(msg):
    
    #Esta parte del codigo sirve para cargar los datos almacenados en el archivo intents.json
    #Recordar: El archivo .json es la base de datos inicial con la cual se alimentará el sistema 
    # y en el cual se pueden agregar nuevas formas de saludar, responder, categorias etc. 
    import json
    with open('intents.json') as file:
        data = json.load(file)

    #Con el bloque siguiente lo que hacemos es obtener cada una de las palabras que tenemos en el 
    #archivo .json y convertirlas en el lenguaje natural, adicionalmente obtengo las catergorias. 


    try:
        with open("data.pickle", "rb") as f:
            words, labels, training, output = pickle.load(f)

        model.load("model.tflearn")       


    except:

        #En caso de un error se ejectuta por aqui. 


        words=[] #Palabras sin deferenciar la frase a la que pertenecen 
        labels=[] #Titulos, legendas.
        docs_x=[]
        docs_y=[]

        #Con este ciclo for estoy recorriendo todo el archivo json y tomando cada una de las frases para 
        #convertirlas en palabras. 

        #Con ese for llenare la variable que guarda las palabra
        for intents in data['intents']:
            for patterns in intents['patterns']:
                wrds = nltk.word_tokenize(patterns) #Convierte una frase a un conjunto de palabras
                words.extend(wrds)
                docs_x.append(wrds)
                docs_y.append(intents["tag"])


                if intents['tag'] not in labels:
                    labels.append(intents['tag'])


        #La informacion y los codigos contenidos en esta celda sirven para recorrer todas las palabras extraidas
        #del archivo .json y convertirlas en el lenguaje natural. Adicionalmente con la funcion list y sorted 
        #logramos eliminar las palabras repetidas y ordenarlas. 


        words=[stemmer.stem(w.lower()) for w in words if w != "?"]


        words = sorted(list(set(words))) #Organizando el conjunto de paralabras de forma no repetiva y ordenada.

        labels = sorted(labels)



        #['greeting', 'goodbye', 'thanks', 'hours', 'payments', 'opentoday']


        #A continuacion se crean dos variables llamadas training y output

        #Deben asemejar a training con las palabras osea words.
        #Deben asemejar a output con las categorias osea labels.

        training=[]
        output=[]

        out_empty = [0 for _ in range (len(labels))]

        #Este ciclo for se encarga de analizar todas y cada una de las palabras en todas y cada una de las frases


        for x, doc in enumerate(docs_x):
            bag = []

            wrds=[stemmer.stem(w.lower()) for w in doc]

            for w in words:
                if w in wrds:
                    
                    bag.append(1)
                else:
                    bag.append(0)
                    
                output_row = out_empty[:]
                output_row[labels.index(docs_y[x])] = 1

                training.append(bag)
                output.append(output_row)



        #Todo el codigo anterior es necesario para llegar a las dos 
        #variables "Finales" que alimentaran el sistema de machine
        #Learning llamadas training y output las cuales formaran 
        #parte de la capa de alimentacion. 

        training = np.array(training) #Contiene la informacion preparada con la cual se va a alimentar el sistema referentes a las palabras
        output = np.array(output) #Contiene la informacion preparada con la cual se va a alimentar el sistema referente a la categorizacion

        with open ("data.pickle", "wb") as f:
            pickle.dump((words, labels, training, output), f)


        #tensorflow.reset_default_graph() #Es la primera vez que utilizo la libreria tensorflow en el codigo y estoy utilizando
        #una funcion de esa libreria llamada reset_default_graph

        tensorflow.compat.v1.reset_default_graph()

        #Con esta linea estoy creando mi primera capa o capa 0 o capa de alimentacion
        net = tflearn.input_data(shape=[None, len(training[0])]) 


        #Con esta linea estoy creando mi primera capa de red neuronal Circulos negros
        net = tflearn.fully_connected(net, 8)


        #Con esta linea estoy creando mi segunda capa de red neuronal Circulos rojos
        net = tflearn.fully_connected(net, 8)


        #Continuacion 
        #Capa de decisión Circulos verdes

        #Otro modelo de regresion es sigmoid
        net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
        net = tflearn.regression(net)

        #Esta linea se encarga de construir el modelo final a partir de las especificaciones anteriores
        model = tflearn.DNN(net)

        try:
            model.load("model.tflearn")
        except:       

            #Hasta el momento hemos configurado nuestro modelo, es hora de entrenarlo con nuestros datos. 
            #Para eso usaremos las siguientes lineas de codigo

            model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
            model.save("model.tflearn")
        
        results = model.predict([bag_of_words(msg, words)])
       
        
        results_index = np.argmax(results) #La funcion argmax obtiene la probabilidad mas alta.
        
        
        #Me devuelve el numero de la posicion donde se encuentra la probabilidad mas alta.
        
        tag = labels[results_index]

        #Finalmente ingreso al archivo json particularmente a la categoria elegida por el modelo
        #y me quedo con las respuestas correspondientes. 
        for tg in data["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses']

        #escogemos una respuesta al azar
        return (random.choice(responses))
    
    

def send():
    msg=EntryBox.get("1.0", 'end-1c').strip()
    EntryBox.delete("0.0", END)
    
    res=chatbot_response(msg) #Aqui posteriormente crearemos una funcion
    
    if msg != '':
        ChatLog.config(state=NORMAL)
        ChatLog.insert(END, "You: "+msg+'\n\n')
        ChatLog.config(foreground="black", font=("Verdana",12))
        
        ChatLog.insert(END, "ChatBOT: "+res+'\n\n')

        ChatLog.config(state=DISABLED)
        ChatLog.yview(END)        
    

#Con esta linea estoy limitando la geometria de la ventana
#Al cambiar los False por True habilito la redimension en alguno de las direcciones 
base.resizable(width=FALSE, height=FALSE)


#Con esta linea estoy creando el cuadro blanco donde se imprimen los mensajes
#Hace referencia al historial de los mensajes. 
ChatLog = Text(base, bd=0, bg="white", height="8", width="50", font="Arial")
ChatLog.config(foreground="black", font=("Verdana", 12))

#Inserto el primer mensaje que debe contener este recuadro
ChatLog.insert(END, "SALUDOS BIENVENIDO"+'\n\n')
ChatLog.place(x=6, y=6, height=386, width=370)

scrollbar = Scrollbar(base, command=ChatLog.yview, cursor = "heart")
ChatLog['yscrollcommand']=scrollbar.set
scrollbar.place(x=376, y=6, height=386)
ChatLog.config(state=DISABLED)

EntryBox=Text(base, bd=0, bg="white", width="29", height="5", font="Arial")
EntryBox.place(x=6, y=401, height=90, width=265)

SendButton = Button(base, font=("Verdana", 12, 'bold'), text="Send", width="9",
                   height=5, bd=0, bg="blue", activebackground="gold", 
                    fg='#ffffff', command=send)
SendButton.place(x=282, y=401, height=90)

base.bind('<Return>', lambda event:send())
base.mainloop()

