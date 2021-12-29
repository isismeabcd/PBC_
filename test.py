import tkinter as tk
import tkinter.font as tkFont
from PIL import ImageTk
import matplotlib.pyplot as pyplot
import os
import PBC80.preprocessor as pp
from PBC80.preprocessor import KBestSelector
from PBC80.classification import DecisionTree
from PBC80.performance import KFoldClassificationPerformance
from tkinter import *
from tkinter import constants
from tkinter.simpledialog import *
import PBC80.model_drawer as md
from IPython.display import Image, display
import pandas as pd

class Plotter(tk.Frame):

  def __init__(self, master = None):
    tk.Frame.__init__(self, master) 
    self.grid()
    self.createWidgets()
    
  def createWidgets(self):
    f = tkFont.Font(size = 16, family = "Courier New")
    self.lblX = tk.Label(self, text = "請輸入檔名:", height = 1, width = 10, fg = "blue4",font = f)
    self.txtX = tk.Text(self, height = 1, width = 40, font = f)
    self.btnLoad = tk.Button(self, text = "計算!", height = 1, width = 5, command = self.clickBtnLo, font = f)
    self.btnLoad2 = tk.Button(self, text = "畫樹!", height = 1, width = 5, command = self.clickBtnLo2, font = f)
    self.cvsMain = tk.Canvas(self, width = 800, height = 600, bg="light green")
	
    self.lblX.grid(row = 2, column = 0, sticky = tk.E)
    self.txtX.grid(row = 2, column = 1, sticky = tk.NE + tk.SW)
    self.btnLoad.grid(row = 4, rowspan = 2, column = 2, sticky = tk.NE + tk.SW)
    self.btnLoad2.grid(row = 7, rowspan = 2, column = 2, sticky = tk.NE + tk.SW)
    self.cvsMain.grid(row = 9, column = 0, columnspan = 3, sticky = tk.NE + tk.SW)
    self.cvsMain.create_text(100, 30, text='模型計算效能:', fill="black",font="Times 20 italic bold")
    self.cvsMain.create_text(100, 210, text='決策樹在哪:', fill="black",font="Times 20 italic bold")

  def clickBtnLo(self):
    f = self.txtX.get("1.0", tk.END)
    self.runmodel(f)
    
  def clickBtnLo2(self):
    f = self.txtX.get("1.0", tk.END)
    self.drawtree(f)
    
  def drawtree(self,f):     
    self.cvsMain.create_text(400, 240, text='很遺憾我不會轉圖檔QWQ', fill="black",font="Times 20 italic bold")

    

  def runmodel(self,f): 
    f=f[0:-1]
    dataset = pp.dataset(file=f)
    X, Y = pp.decomposition(dataset, x_columns=[i for i in range(1, 23)], y_columns=[0])
    X = pp.onehot_encoder(X, columns=[i for i in range(22)], remove_trap=True)
    Y, Y_mapping = pp.label_encoder(Y, mapping=True)
    selector = KBestSelector(best_k="auto")
    X = selector.fit(x_ary=X, y_ary=Y, verbose=True, sort=True).transform(x_ary=X)
    X_train, X_test, Y_train, Y_test = pp.split_train_test(x_ary=X, y_ary=Y)
    classifier = DecisionTree()
    Y_pred = classifier.fit(X_train, Y_train).predict(X_test)
    K = 10
    kfp = KFoldClassificationPerformance(x_ary=X, y_ary=Y, classifier=classifier.classifier, k_fold=K)
    out="{} Folds Mean Accuracy: {}\n{} Folds Mean Recall: {}\n{} Folds Mean Precision: {}\n{} Folds Mean F1-Score: {}" .format(K, kfp.accuracy(),K, kfp.recall(),K,kfp.precision(),K, kfp.f_score())
    self.cvsMain.create_text(400, 120, text=out, fill="black",font="Times 20 italic bold")
    
  
pl = Plotter()
pl.master.title("My Plotter")
pl.mainloop()