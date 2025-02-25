import tkinter as tk
from tkinter import END
my_w = tk.Tk()
my_w.geometry("800x500")  # Size of the window
my_w.title("www.plus2net.com")  # Adding a title
font1=('Times',34,'normal')

l1 = tk.Label(my_w,  text='Meter', width=10,font=font1 )  # Label
l1.grid(row=0,column=0,padx=10,pady=10)
m1_var=tk.DoubleVar()
m1 = tk.Entry(my_w,width=10,bg='yellow',font=font1,textvariable=m1_var)
m1.grid(row=0,column=1,padx=10)

l2 = tk.Label(my_w,  text='Feet', width=10,font=font1 )  # Label
l2.grid(row=1,column=0,padx=10,pady=10)

f1_var=tk.DoubleVar()
f1 = tk.Entry(my_w,width=10,bg='yellow',font=font1,textvariable=f1_var)
f1.grid(row=1,column=1,padx=10)

l3 = tk.Label(my_w,  text='Kadi', width=10,font=font1 )  #  Label
l3.grid(row=2,column=0,padx=10,pady=10)

k1_var=tk.DoubleVar()
k1 = tk.Entry(my_w,width=10,bg='yellow',font=font1,textvariable=k1_var) # text box
k1.grid(row=2,column=1,padx=10)

def my_upd1(*args): # feet is entered
    in_meeter=round(f1_var.get()*0.3048,2)
    m1_var.set(in_meeter)
    k1_var.set(round(f1_var.get()*0.66,2))
def my_upd2(*args): # meeter is entered
    in_feet=m1_var.get()*3.2808
    f1_var.set(round(in_feet,2))
    k1_var.set(round(in_feet/0.66,2))
def my_upd3(*args): # kadi is entered
    in_feet=k1_var.get()*0.66
    m1_var.set(round(in_feet*0.3048,2))
    f1_var.set(round(in_feet,2))

f1.bind("<FocusOut>",my_upd1)
m1.bind("<FocusOut>",my_upd2)
k1.bind("<FocusOut>",my_upd3)

m1.bind("<FocusIn>",lambda x: m1.select_range(0,tk.END))
f1.bind("<FocusIn>",lambda x: f1.select_range(0,tk.END))
k1.bind("<FocusIn>",lambda x: k1.select_range(0,tk.END))
my_w.mainloop()  # Keep the window open