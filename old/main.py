#! /usr/bin/env python3
# -*- coding: utf-8 -*-

from generate import Generator
from plan import Planner

import tkinter as tk
from tkinter import ttk
from tkinter import Entry
from tkinter import StringVar

# if __name__ == '__main__':
#     planner = Planner()
#     generator = Generator()
#     while True:
#         hints = input("Type in hints >> ")
#         keywords = planner.plan(hints)
#         print("Keywords: " + ' '.join(keywords))
#         poem = generator.generate(keywords)
#         print("Poem generated:")
#         for sentence in poem:
#             print(sentence)


def clickme():
    hints = e1.get()
    keywords = planner.plan(hints)
    print(keywords)
    poem = generator.generate(keywords)
    print(poem)
    aLabel.configure(text=keywords)
    bLabel.configure(text=poem)

if __name__ == '__main__':
    
    planner = Planner()
    generator = Generator()
    
    win=tk.Tk()
    win.title('poem')
    win.geometry("400x200")
    
    v1 = StringVar()
    e1 = Entry(win, textvariable = v1)
    e1.pack()
    
    action=ttk.Button(win,text='生成',command=clickme)
    action.pack()
    
    #show result
    aLabel=ttk.Label(win,text="Waiting for keyword",width=80)
    aLabel.pack()
    
    bLabel=ttk.Label(win,text="Waiting for poem", width=80)
    bLabel.pack()
    
    win.mainloop()
