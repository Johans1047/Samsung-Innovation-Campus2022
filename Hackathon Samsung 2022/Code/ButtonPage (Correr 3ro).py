import telebot
from telebot.types import ReplyKeyboardMarkup
from telebot.types import ForceReply
from telebot.types import ReplyKeyboardRemove
import nltk
from nltk.stem.lancaster import LancasterStemmer
from dotenv import load_dotenv
from tkinter import *
import datetime
from datetime import *
from PIL import Image, ImageTk
import pandas as pd
import numpy as np
import tkinter
import re
import tensorflow
import tflearn
import random
import pickle
import discord
import os
import json
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# stemmer=SnowballStemmer('spanish')
stemmer = LancasterStemmer()


# ________________ Generar citas ________________

def random_date(start, end, fmt):
    s = datetime.strptime(start, fmt)
    e = datetime.strptime(end, fmt)
    delta = e - s

    appointment = s + timedelta(days=(random.random() * delta.days))
    return appointment.strftime('%d/%m/%Y')


def gen_cita():
    dateObj = datetime.now()  # import date
    todayDate = dateObj.strftime('%d/%m/%Y')  # give format
    topDateObj = dateObj + timedelta(days=10)
    topDate = topDateObj.strftime('%d/%m/%Y')
    cita = random_date(todayDate, topDate, "%d/%m/%Y")

    return cita


# print("fecha de hoy ", todayDate)
# print("fecha tope ", topDate)
print("\n\nCita generada por el sistema: ", gen_cita())


def upload_appointment(appointment):
    df = pd.read_csv('informe.csv')
    df.loc[len(df)] = [appointment]
    df.to_csv('informe.csv', index=False)


# _____________________________________________ PYTHON WINDOW ____________________________________________________
def chatbot_response(msg):
    import json
    with open('intents.json') as file:
        data = json.load(file)

    try:
        with open("data.pickle", "rb") as f:
            words, labels, training, output, data = pickle.load(f)

        with open('intents.json') as file:

            data2 = json.load(file)

        if data != data2:
            raise Exception("El Archivo json ha cambiado")
        model.load("model.tflearn")

    except:
        print("Estoy dentro del EXCEPT")

        words = []  # Palabras sin deferenciar la frase a la que pertenecen
        labels = []  # Titulos, legendas.
        docs_x = []
        docs_y = []

        for intents in data['intents']:
            for patterns in intents['patterns']:
                wrds = nltk.word_tokenize(patterns)  # Convierte una frase a un conjunto de palabras
                words.extend(wrds)
                docs_x.append(wrds)
                docs_y.append(intents["tag"])

                if intents['tag'] not in labels:
                    labels.append(intents['tag'])

        words = [stemmer.stem(w.lower()) for w in words if w != "?"]

        words = sorted(list(set(words)))  # Organizando el conjunto de paralabras de forma no repetiva y ordenada.

        labels = sorted(labels)

        training = []
        output = []

        out_empty = [0 for _ in range(len(labels))]

        # Este ciclo for se encarga de analizar todas y cada una de las palabras en todas y cada una de las frases

        for x, doc in enumerate(docs_x):
            bag = []

            wrds = [stemmer.stem(w.lower()) for w in doc]

            for w in words:
                if w in wrds:
                    bag.append(1)
                else:
                    bag.append(0)

                output_row = out_empty[:]
                output_row[labels.index(docs_y[x])] = 1

                training.append(bag)
                output.append(output_row)

        training = np.array(
            training)  # Contiene la informacion preparada con la cual se va a alimentar el sistema referentes a las palabras
        output = np.array(
            output)  # Contiene la informacion preparada con la cual se va a alimentar el sistema referente a la categorizacion

    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output, data), f)

    tensorflow.compat.v1.reset_default_graph()

    net = tflearn.input_data(shape=[None, len(training[0])])
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, 8)

    net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
    net = tflearn.regression(net)

    model = tflearn.DNN(net)

    try:
        model.load("model.tflearn")
    except:
        model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
        model.save("model.tflearn")

    results = model.predict([bag_of_words(msg, words)])
    results_index = np.argmax(results)  # La funcion argmax obtiene la probabilidad mas alta.

    tag = labels[results_index]
    print(tag)
    print(labels)
    print("esto es antes del for")
    for tg in data["intents"]:

        if tg['tag'] == "Appointment" and tg['tag'] == tag:
            appointment = gen_cita()
            print(tg['tag'] + "==" + 'Appointment')
            responses = ("cita generada: ", appointment)
            upload_appointment(appointment)
            print("ENTRO AL OTRO IF")
        else:
            if tg['tag'] == tag and tg['tag'] != "intents":
                print(tg['tag'] + "==" + 'Appointment')
                responses = tg['responses']
                print("ENTRO AL IF")

        # if tg['tag'] == tag:

    # else:
    #   responses = ("cita generada: ", gen_cita())

    return (random.choice(responses))

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]
    s_words = nltk.word_tokenize(s)  # Convierto la frase ingresada por el usuario en palabras
    s_words = [stemmer.stem(word.lower()) for word in s_words]
    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return np.array(bag)

def ventana():
    print("ENTRA AL CHATBOT")

    def send(Chatlog, EntryBox):
        msg = EntryBox.get("1.0", 'end-1c').strip()
        print("mensaje captado: ", EntryBox.get("1.0", "end-1c"))

        print("\nborrando....")
        EntryBox.delete("0.0", END)
        print("mensaje borrado: ", EntryBox.get("1.0", "end-1c"))

        res = chatbot_response(msg)

        if msg != "":
            ChatLog.tag_configure('tag-right', justify='right')
            ChatLog.tag_configure('tag-left', justify='left')
            ChatLog.config(state=NORMAL)

            frame1 = Frame(ChatLog, bg="#d0ffff")
            Label(frame1, text=msg, font=("Verdana", 10), bg="#d0ffff").grid(row=0, column=0, sticky="e",
                                                                             padx=5, pady=5)
            Label(frame1, text=datetime.now().strftime("%H:%M"), font=("Arial", 7), bg="#d0ffff").grid(row=1, column=0,
                                                                                                       sticky="w")
            # ChatLog.insert(END, "You: " + msg + " \n\n")
            # ChatLog.config(foreground="black", font=("Verdana", 12))
            # ChatLog.insert(END, "ChatBot: " + res + " \n\n")
            frame2 = Frame(ChatLog, bg="#ffffd0")

            Label(frame2, text=res, font=("Verdana", 10), bg="#ffffd0").grid(row=0, column=0, sticky="e",
                                                                             padx=5, pady=5)
            Label(frame2, text=datetime.now().strftime("%H:%M"), font=("Arial", 7), bg="#ffffd0").grid(row=1, column=0,
                                                                                                       sticky="e")
            # ChatLog.config(state=DISABLED)
            # ChatLog.yview(END)

            ChatLog.insert('end', '\n', 'tag-left')
            ChatLog.window_create('end', window=frame1)

            ChatLog.insert('end', '\n ', 'tag-right')
            ChatLog.window_create('end', window=frame2)

            ChatLog.tag_add("tag-right", "end-1c linestart", "end-1c lineend")
            ChatLog.config(state=DISABLED)
            ChatLog.yview(END)
            """ChatLog.config(state=NORMAL)
            ChatLog.insert(END, "You: " + msg + " \n\n")
            ChatLog.config(foreground="black", font=("Verdana", 12))
            ChatLog.insert(END, "ChatBot: " + res + " \n\n")
            ChatLog.config(state=DISABLED)
            ChatLog.yview(END)"""

    base = Tk()
    base.title("Chatbot (Atención al usuario)")
    base.geometry("400x500")
    base.config(bg="#00a9e0")
    base.resizable(width=FALSE, height=FALSE)  # Mantiene fija la ventana.

    ChatLog = Text(base, bd=0, bg="white", width=8, height="50", font="Arial")
    ChatLog.config(foreground="Black", font=("Verdana", 10))  # foreground cambia el color de la letra
    ChatLog.insert(END, "Bienvenido, Mi nombre es Pythinbot, ¿Como te ayudo?\n\n")
    ChatLog.place(x=6, y=6, height=386, width=370)
    ChatLog.config(state=DISABLED)  # bloquea la entrada de texto(lo hace de solo lectura)

    scrollbar = Scrollbar(base, command=ChatLog.yview(), cursor="heart")
    ChatLog['yscrollcommand'] = scrollbar.set
    scrollbar.place(x=376, y=6, height=386)

    EntryBox = Text(base, bg="white", height="5", width=29, font="Arial")
    EntryBox.place(x=6, y=401, height=90, width=265)

    SendButton = Button(base, font=("Verdana", 12, "bold"), text="Send", height="5", width="9",
                        bd=3, bg="white", activebackground="#0689d8", fg='#0689d8', command=send)
    SendButton.place(x=282, y=401, height=90)

    base.bind('<Return>', lambda event: send(ChatLog, EntryBox))

    #  Primer parameter: posición
    #  ChatLog.grid(column=0, row=0)

    base.mainloop()


def pytk():
    buttonPage.destroy()

    # _______Validaciones______
    def es_nombre(nombre):
        nombre = nombre.replace(' ', '')

        if nombre.isalpha() == False:
            nombre = ""
            raise ValueError()

    # ___ Telefono ___
    def validar_telefono(numero):
        patron = re.compile(r'^\d{4}\d{4}$')

        return patron.match(numero)

    # ____ Correo ____
    def validar_correo(correo):
        print("está validando")
        patron = r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$'
        if re.match(patron, correo):
            print("validó el correo")
            return True
        else:
            return False

    # ______ Guardar el archivo en csv _________

    def save_to_csv(email, name, tel, subscribed):
        dict = {'Name': '', 'eMail': '', 'Phone': 0, 'Subscribed': ''}

        dict['Name'] = name
        dict['eMail'] = email
        dict['Phone'] = tel
        dict['Subscribed'] = subscribed

        try:
            df = pd.DataFrame(dict, index=[0])
            df.to_csv("informe.csv", sep=";", mode='a', index=False, header=False)
            pass

        except:
            # df = pd.DataFrame(dict, index=[0])
            # df.to_csv("informe.csv", sep=";", index=False)
            pass

    # _________ validar correo y telefono ________
    def validate():
        email = correo_in.get("1.0", "end-1c").strip()
        name = nombre_in.get("1.0", "end-1c").strip()
        tel = tel_in.get("1.0", "end-1c").strip()

        if checkvar.get() == 0:
            subscribed = 'No'
        else:
            subscribed = 'yes'

        if validar_correo(email):
            print("CORREO VALIDO")
            validate_mail = True
        else:
            tkinter.messagebox.showerror(message="Correo inválido", title="ERROR")
            print("CORREO INVALIDO")
            validate_mail = False
        if validar_telefono(tel):
            print("NUMERO VALIDO")
            validate_phone = True
        else:
            print("NUMERO INVÁLIDO")
            tkinter.messagebox.showerror(
                message="Numero de teléfono inválido, no incluyas letras ni caracteres especiales",
                title="ERROR")
            validate_phone = False

        if (validate_phone and validate_mail):
            print("datos guardados...")
            save_to_csv(email, name, tel, subscribed)  # Guarda
            log_in.destroy()  # Cierra la ventana del LOGIN
            ventana()  # Corre el chatbot
            print("\n○\nLanzando chatbot...")

    names = []
    emails = []
    phones = []
    subs = []
    informe = pd.DataFrame()
    headers = ["Name", "eMail", "Number", "Subscribed"]
    # __________________________ front end __________________________
    log_in = Tk()
    log_in.geometry("550x500")
    log_in.title("Inicia sesión")
    log_in.resizable(width=False, height=False)

    # ______ Imagen Background _______
    imagen = PhotoImage(file="./Imagenes/SamsungBackground.png")
    background_label = Label(log_in, image=imagen)
    background_label.place(x=0, y=0, relwidth=1, relheight=1)
    texto1 = "introduce tu nombre"

    # ______ Label que indica al usuario que ingrese el nombre _______

    nombre_lab = Label(log_in, compound=CENTER, text=texto1, image=imagen, padx=100, pady=70, takefocus=True)
    nombre_lab.config(foreground="white", font=("Montserrat, sans-serif", "12", "bold"))
    nombre_lab.place(x=100, y=70, height=25, width=170)

    # ______ Usuario ingresa el nombre _______
    nombre_in = Text(log_in, bd=0, bg="white", width=1, height="5", font="Arial")
    nombre_in.config(foreground="black", font=("Verdana", 12))
    nombre_in.place(x=100, y=100, height=25, width=350)

    # ______ Label que indica al us-uario que ingrese el correo _______

    correo_lab = Label(log_in, width=1, height="5", font="Arial", compound=CENTER, image=imagen,
                       text="Introduce tu correo electronico")
    correo_lab.config(foreground="white", font=("Montserrat, sans-serif", 12, "bold"), bd=1, background="ghost white")
    correo_lab.place(x=100, y=145, height=25, width=240)

    w_mail = Label(log_in, width=1, height="5", font="Arial", compound=CENTER, image=imagen,
                   text="(nombre@dominio.com)")
    w_mail.config(foreground="white", font=("Montserrat, sans-serif", 7))
    w_mail.place(x=350, y=150, height=20, width=100)

    # ______ Usuario ingresa el correo _______
    correo_in = Text(log_in, bd=0, bg="white", width=1, height="5", font="Arial")
    correo_in.config(foreground="black", font=("Verdana", 12))
    correo_in.place(x=100, y=175, height=25, width=350)

    # ______ Label que indica al usuario que ingrese el numero de telefono _______

    tel_lab = Label(log_in, bd=0, width=1, height="5", font="Arial", compound=CENTER, image=imagen,
                    text="Introduce tu numero de telefono")
    tel_lab.config(foreground="white", font=("Montserrat, sans-serif", 12, "bold"))
    tel_lab.place(x=100, y=220, height=25, width=253)

    w_tel = Label(log_in, bd=0, bg="ghost white", width=1, height="5", font="Arial", compound=CENTER, image=imagen,
                  text="(xxxxxxxx)")
    w_tel.config(foreground="white", font=("Montserrat, sans-serif", 8))
    w_tel.place(x=370, y=220, height=25, width=60)

    # ______ Usuario ingresa el numero de telefono _______
    tel_in = Text(log_in, bd=0, bg="white", width=1, height="5", font="Arial")
    tel_in.config(foreground="black", font=("Verdana", 12))
    tel_in.place(x=100, y=250, height=25, width=350)

    # __________ SMI Logo ___________

    samsung_logo = ImageTk.PhotoImage(Image.open("./Imagenes/Samsung.png"))
    samsung_label = Label(log_in, image=samsung_logo, background="#0689d8", compound=CENTER)
    samsung_label.place(x=100, y=350)

    # _______ Checkvar ___________

    checkvar = IntVar()
    check_btn = tkinter.Checkbutton(log_in, text="¿Deseas estar suscribirte a nuestro portal?",
                                    variable=checkvar)
    check_btn.config(background="#0689d8", bd=3, font=("Montserrat, sans-serif", 9), foreground="white")
    check_btn.place(x=95, y=300)

    # _______ Botón de enviar ___________
    sendButton = Button(log_in, font=("Montserrat, sans-serif", 12, "bold"),
                        text="Enviar", height="5", width="9",
                        bd=3, bg="#0689d8", activebackground="#1428a0",
                        fg='white', command=validate)
    sendButton.place(x=350, y=350, height=90)

    # ______ Crear dataframe para guardar los datos ________

    log_in.mainloop()


# __________________________________________ TELEGRAM BOT _____________________________________________________________

def start_telegram():
    stemmer = LancasterStemmer()
    tag = ''

    bot = telebot.TeleBot("5866968877:AAHxSOvDb4qvINfXcJqzCl8LDnfHIBP64xE", parse_mode=None)
    usuarios = {}

    @bot.message_handler(commands=['start', 'help'])
    def send_welcome(message):
        markup = ReplyKeyboardRemove()
        bot.reply_to(message, "Hola, Bienvenido por favor usa el comando /alta para comenzar ", reply_markup=markup)

    @bot.message_handler(func=lambda m: True)
    @bot.message_handler(commands=['alta'])
    def cmd_alta(message):
        markup = ForceReply()
        msg = bot.send_message(message.chat.id, "Alta fidelidad, por favor dinos tu nombre", reply_markup=markup)
        bot.register_next_step_handler(msg, preguntar_edad)



    def cmd_cita(message):
        markup = ForceReply()
        msg = bot.send_message(message.chat.id, "Su cita", reply_markup=markup)

    def preguntar_edad(message):
        usuarios[message.chat.id] = {}
        usuarios[message.chat.id]["nombre"] = message.text
        markup = ForceReply()
        msg = bot.send_message(message.chat.id, "Danos tu edad", reply_markup=markup)
        bot.register_next_step_handler(msg, preguntar_telefono)

    def preguntar_telefono(message):
        usuarios[message.chat.id]["edad"] = message.text
        markup = ForceReply()
        edad = message.text
        print(edad)
        if not edad.isdigit():
            markup = ForceReply()
            msg = bot.send_message(message.chat.id, "Error solo se permiten numeros")
            bot.register_next_step_handler(msg, preguntar_telefono)
        else:
            msg = bot.send_message(message.chat.id, "Nos facilitas tu numero", reply_markup=markup)
            bot.register_next_step_handler(msg, preguntar_correo)

    def preguntar_correo(message):
        usuarios[message.chat.id]["numero"] = message.text
        markup = ForceReply()
        numero = message.text
        if not numero.isdigit():
            markup = ForceReply()
            msg = bot.send_message(message.chat.id, "Error solo se permiten numeros")
            bot.register_next_step_handler(msg, preguntar_telefono)
        else:
            msg = bot.send_message(message.chat.id, "Nos facilitas tu correo", reply_markup=markup)
            bot.register_next_step_handler(msg, menu)

    def menu(message):
        usuarios[message.chat.id]["correo"] = message.text
        markup = ReplyKeyboardMarkup(one_time_keyboard=True,
                                     input_field_placeholder="Pulse un boton",
                                     resize_keyboard=True)
        markup.add("Sacar Cita", "Comprar Producto", "Cotizacion", "Consula")
        msg = bot.send_message(message.chat.id, "Que desea realizar??", reply_markup=markup)
        bot.register_next_step_handler(msg, datos_cu)
        print(usuarios)
        fromTelegram_save_to_csv(usuarios)

    def fromTelegram_save_to_csv(usuarios):
        try:
            df = pd.DataFrame(usuarios)
            print(df)
            df.to_csv("informeTelegram.csv", sep=";", mode='a', index=False, header=False)
            pass
        except:
            pass


    def datos_cu(message):
        markup = ReplyKeyboardRemove()
        print(usuarios)
        if message.text == "Sacar Cita":
            msg = bot.send_message(message.chat.id, "Listo que quiere ver si cita use el comando /quiero uan cita",
                                   reply_markup=markup)
            msg = bot.send_message(message.chat.id, chatbot_response(msg), reply_markup=markup)
        elif message.text == "Comprar Producto":
            msg = bot.send_message(message.chat.id, "Que desea comprar", reply_markup=markup)
            bot.register_next_step_handler(msg, compra)
        elif message.text == "Consula":
            msg = bot.send_message(message.chat.id, "Que desea consultar?", reply_markup=markup)
            bot.register_next_step_handler(msg, consulta)
        elif message.text == "Cotizacion":
            msg = bot.send_message(message.chat.id, "Que desea cotizar?", reply_markup=markup)
            bot.register_next_step_handler(msg, menu)
        else:
            msg = bot.send_message(message.chat.id, "Solo se por favor selecione un recuadro", reply_markup=markup)
            bot.register_next_step_handler(msg, datos_cu)

    def consulta(message):
        text = message.text
        msg = bot.send_message(message.chat.id, chatbot_response(text))

    def cita_tele(message):
        text = message.text
        msg = bot.send_message(message.chat.id, chatbot_response(text))

    def compra(message):
        text = message.text
        msg = bot.send_message(message.chat.id, chatbot_response(text))

    def cotizar(message):
        text = message.text
        msg = bot.send_message(message.chat.id, chatbot_response(text))
    bot.infinity_polling()

    # _________________________________________ LINK WITH DISCORD __________________________________________________________


def run_discord():
    buttonPage.destroy()
    stemmer = LancasterStemmer()
    tag = ''
    with open('intents.json') as json_data:
        intents = json.load(json_data)

    words = []
    classes = []
    documents = []
    ignore_words = ['?', '¿']
    print("Looping through the Intents to Convert them to words, classes, documents and ignore_words.......")

    for intent in intents['intents']:
        for pattern in intent['patterns']:
            # tokenize each word in the sentence
            w = nltk.word_tokenize(pattern)
            # add to our words list
            words.extend(w)
            # add to documents in our corpus
            documents.append((w, intent['tag']))
            # add to our classes list
            if intent['tag'] not in classes:
                classes.append(intent['tag'])

    print("Stemming, Lowering and Removing Duplicates.......")
    words = [stemmer.stem(w.lower()) for w in words if w not in ignore_words]
    words = sorted(list(set(words)))

    classes = sorted(list(set(classes)))

    print(len(documents), "documents")
    print(len(classes), "classes", classes)
    print(len(words), "unique stemmed words", words)
    print("Creating the Data for our Model.....")
    training = []
    output = []
    print("Creating an List (Empty) for Output.....")
    output_empty = [0] * len(classes)

    print("Creating Traning Set, Bag of Words for our Model....")
    for doc in documents:
            # initialize our bag of words
        bag = []
            # list of tokenized words for the pattern
        pattern_words = doc[0]
            # stem each word
        pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]
            # create our bag of words array
        for w in words:
            bag.append(1) if w in pattern_words else bag.append(0)

            # output is a '0' for each tag and '1' for current tag
        output_row = list(output_empty)
        output_row[classes.index(doc[1])] = 1

        training.append([bag, output_row])

    print("Shuffling Randomly and Converting into Numpy Array for Faster Processing......")
    random.shuffle(training)
    training = np.array(training)

    print("Creating Train and Test Lists.....")
    train_x = list(training[:, 0])
    train_y = list(training[:, 1])
    net = tflearn.input_data(shape=[None, len(train_x[0])])
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
    net = tflearn.regression(net)
    print("Training....")
    model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')

    model.load('model.tflearn')

    def clean_up_sentence(sentence):
        sentence_words = nltk.word_tokenize(sentence)
        sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
        return sentence_words

    def bow(sentence, words, show_details=False):
        sentence_words = clean_up_sentence(sentence)
        bag = [0] * len(words)
        for s in sentence_words:
            for i, w in enumerate(words):
                if w == s:
                    bag[i] = 1
                    if show_details:
                        print("found in bag: %s" % w)
        return (np.array(bag))

    ERROR_THRESHOLD = 0.25
    print("ERROR_THRESHOLD = 0.25")

    def classify(sentence):
        results = model.predict([bow(sentence, words)])[0]
        results = [[i, r] for i, r in enumerate(results) if r > ERROR_THRESHOLD]
        results.sort(key=lambda x: x[1], reverse=True)
        return_list = []
        for r in results:
            return_list.append((classes[r[0]], r[1]))  # Tuppl -> Intent and Probability
            return return_list

    def response(sentence, userID='123', show_details=False):
        results = classify(sentence)
        if results:
            while results:
                for i in intents['intents']:
                    if i['tag'] == results[0][0]:
                        return i

                results.pop(0)

    load_dotenv()  # load all the variables from the env file
    bot = discord.Bot()

    @bot.event
    async def on_ready():
        print(f"{bot.user} is ready and online!")

    @bot.event
    async def on_message(message):
        global tag

        if tag != 'greeting':
            if message.author != bot.user:
                    # SENDS BACK A MESSAGE TO THE CHANNEL.
                await message.channel.send(random.choice(response(message.content)['responses']))
                tag = response(message.content)['tag']
            else:
                nombre = ''
                for i in message.content:
                    if (i.isnumeric()) == False:
                        nombre += i
            if message.author != bot.user:
                    await message.channel.send(('Bienvenido ' + nombre))
                    tag = response(message.content)['tag']

    @bot.slash_command(name="hello", description="Say hello to the bot")
    async def hello(ctx):
        await ctx.respond("Hey!")

    bot.run(os.getenv('TOKEN'))

# __________________________________________ BUTTON WINDOW _____________________________________________________________

buttonPage = Tk()
buttonPage.config(height=250, width=500)
buttonPage.title("Bievenido")
buttonPage.resizable(width=False, height=False)

# _______________ ICONS ___________________
photo_discord = PhotoImage(file="./Imagenes/discordd.png")
photo_telegram = PhotoImage(file="./Imagenes/telegramm.png")
photo_python = PhotoImage(file="./Imagenes/python.png")
background = PhotoImage(file="./Imagenes/FirstScreen_background.png")
photo_powerOn = PhotoImage(file="./Imagenes/poweron.png")
photo_shutDown = PhotoImage(file="./Imagenes/shutdown.png")

# ________________ LABELS __________________
background_label = Label(buttonPage, image=background)
background_label.place(x=0, y=0)

# __________ principal label _______________
label_principal = Label(buttonPage, height=2, width=30,
                        text="¿Donde desea iniciar el chatbot?", bg="#0057b8",
                        foreground="white", font=("Verdana", 12, "bold")
                        )
label_principal.place(x=85, y=50)

# __________ DISCORD LABEL _________________

label_discord = Label(buttonPage, height=1, width=8,
                      text="DISCORD",
                      foreground="white", font=("Verdana", 8, "bold"),
                      background="#0057b8")
label_discord.place(x=90, y=190)

# __________ TELEGRAM LABEL ________________
label_telegram = Label(buttonPage, height=1, width=9,
                       text="TELEGRAM",
                       foreground="white", font=("Verdana", 8, "bold"),
                       background="#0057b8")
label_telegram.place(x=215, y=190)
# __________ PYTHON LABEL  _________________
label_python = Label(buttonPage, height=1, width=9,
                     text="PYTHON",
                     foreground="white", font=("Verdana", 8, "bold"),
                     background="#0057b8")
label_python.place(x=340, y=190)

# ____________ DISCORD BUTTON ____________
iniciar_discord = Button(buttonPage, height=50, width=75, bg="white",
                         foreground="black", text="Discord", image=photo_discord)
iniciar_discord.place(x=85, y=120)

shutDownDiscord_button = Button(buttonPage, height=30, width=30,
                                image=photo_shutDown, bg="white", bd=0)
shutDownDiscord_button.place(x=130, y=215)

powerOnDiscord_button = Button(buttonPage, height=30, width=30,
                               image=photo_powerOn, bg="white", bd=0, command=run_discord)
powerOnDiscord_button.place(x=90, y=215)

# ____________ TELEGRAM BUTTON ____________
iniciar_telegram = Button(buttonPage, height=50, width=75, bg="white", foreground="black", text="Telegram",
                          image=photo_telegram)
iniciar_telegram.place(x=215, y=120)

shutDownTelegram_button = Button(buttonPage, height=30, width=30,
                                 image=photo_shutDown, bg="white", bd=0)
shutDownTelegram_button.place(x=260, y=215)

powerOnTelegram_button = Button(buttonPage, height=30, width=30,
                                image=photo_powerOn, bg="white", bd=0, command=start_telegram)
powerOnTelegram_button.place(x=215, y=215)

# ____________ PYTHON BUTTON ____________
iniciar_interfaz = Button(buttonPage, height=50, width=75,
                          bg="white", foreground="black", text="Interfaz", image=photo_python,
                          command=pytk)
iniciar_interfaz.place(x=340, y=120)

buttonPage.mainloop()
