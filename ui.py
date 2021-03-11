from tkinter import *
from PIL import ImageTk,Image
from tkinter import filedialog
import os
import generate

def choose_file():
    filename = filedialog.askopenfilename(initialdir=os.getcwd(), title="Select image file", filetypes=(("JPG File", "*.jpg"), ("PNG file", "*.png"), ("All files", "*.")))
    entry1.delete(0, 'end')
    entry1.insert(0,str(filename))
    img = Image.open(filename)
    img.thumbnail((550,500))
    img = ImageTk.PhotoImage(img)
    lbl.configure(image=img)
    lbl.image = img
    

root = Tk()
root.title("Image Caption Generator Using Deep Learning")
root.geometry("650x650")

mylabel = Label(root, text = " ", font="24")

def generateCaption(mylabel):
    #filePath = entry1.get()
    #fileNameArr = filePath.split("/")
    #file_name = fileNameArr[len(fileNameArr)-1]
    file_name = entry1.get()
    print(file_name)
    caption = generate.runModel(file_name)

    mylabel = mylabel.config(text=caption)
    #mylabel.update()
    

frm = Frame(root)
frm.pack(side=BOTTOM, padx=10, pady=10)


lbl = Label(root)
lbl.pack()

entry1 = Entry(frm,width =90)

button1 = Button(frm, text = "Select Image",command = choose_file, width=20)

button2 = Button(frm, text="Generate Caption", command= lambda : generateCaption(mylabel), width=20)


entry1.pack()
mylabel.pack()
button1.pack(pady=5)
button2.pack(padx=10, pady=10)


root.mainloop()
