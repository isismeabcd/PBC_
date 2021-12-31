import tkinter as tk
import tkinter.font as tkFont
import PIL
from tkinter import Tk
from tkinter import messagebox
from tkinter.filedialog import askopenfilename
import PBC80.preprocessor as pp
from PBC80.preprocessor import KBestSelector
from PBC80.classification import DecisionTree
from PBC80.performance import KFoldClassificationPerformance
from sklearn import tree
import pydotplus
import os
import pylab
import PBC80.model_drawer as md
from IPython.display import Image, display
from pydotplus import graph_from_dot_data
from PIL import ImageTk
from tkinter import Tk, Canvas
from tkinter import ttk


class maketree(tk.Frame):
    def __init__(self):
        tk.Frame.__init__(self)
        self.grid()
        self.createWidgets()

    #創造不同欄位
    def createWidgets(self):
        f1 = tkFont.Font(size = 16, family = 'Arial')
        f2 = tkFont.Font(size = 20, family = 'Arial')
        f3 = tkFont.Font(size = 28, family = 'Arial')
        f4 = tkFont.Font(size = 16, family = 'Arial')

        radioValue = tk.IntVar()
        radioValue2 = tk.IntVar()
        radioValue3 = tk.IntVar()
    
        intro = '歡迎使用決策樹產生器！\n請選擇需進行資料整理的檔案，並依序設定是否進行離群值剔除、補齊缺失值、降維度後，按下開始鍵，即可生成決策樹以及模型計算效能。'
        self.l_title = tk.Label(self, text = '決策樹產生器', bg = 'PowderBlue', fg = 'CadetBlue', font = f3)
        self.m_intro = tk.Message(self, text = intro, bg = 'Gainsboro', aspect = 1000, font = f1)
        self.l_pickfile = tk.Label(self, text = '請上傳檔案：', fg = 'black', font = f2)
        self.b_pickfile = tk.Button(self, text = '選擇檔案', bg = 'gray', fg = 'black', command = self.clickb_p, font = f2)
        self.l_filecheck = tk.Label(self, text = '您尚未選擇檔案', fg = 'gray', font = f1)
        self.l_setting = tk.Label(self, text = '資料處理設定', bg = 'Gainsboro', fg = 'black', font = f2)
        
        remind = '* 如未進行設定，則預設為：剔除離群值、以KNN Imputer方式補齊缺失值、進行降維處理。'
        self.l_remind = tk.Label(self, text = remind, bg = 'LightCoral', fg = 'Snow', font = f1)
    #第一題：離群值
        self.l_Q1 = tk.Label(self, text = '1. 是否要踢除離群值(outlier)？', font = f4)
        self.Q1_1 = tk.Radiobutton(self, text = '是，以缺失值處理', variable = radioValue, value = 1 , font = f4)
        self.Q1_2 = tk.Radiobutton(self, text = '否，保留離群值', variable = radioValue, value = 2, font = f4)
        self.Q1_check = tk.Button(self, text = '檢視', bg = 'gray', fg = 'black',font = f4,command=self.check_outlier)
        #第二題：缺失值
        self.l_Q2 = tk.Label(self, text = '2. 缺失值替補方式？', font = f4)
        self.Q2_1 = tk.Radiobutton(self, text = '平均值', variable = radioValue2, value = 1 , font = f4)
        self.Q2_2 = tk.Radiobutton(self, text = '中位數', variable = radioValue2, value = 2, font = f4)
        self.Q2_3 = tk.Radiobutton(self, text = '眾數', variable = radioValue2, value = 3 , font = f4)
        self.Q2_4 = tk.Radiobutton(self, text = 'KNN Imputer', variable = radioValue2, value = 4, font = f4)
        self.Q2_check = tk.Button(self, text = '檢視', bg = 'gray', fg = 'black', font = f4)

        #第三題：降維度
        self.l_Q3 = tk.Label(self, text = '3. 是否要將資料降維處理(PCA)？', font = f4)
        self.Q3_1 = tk.Radiobutton(self, text = '是', variable = radioValue3, value = 1 , font = f4)
        self.Q3_2 = tk.Radiobutton(self, text = '否', variable = radioValue3, value = 2, font = f4)
    
        #輸完資料後開始進行處理
        self.b_continue = tk.Button(self, text = '下一步', command = self.click_con, font = f2)
        self.l_title.grid(row = 1, column = 1, columnspan = 30, sticky = tk.SW + tk.NE)
        self.m_intro.grid(row = 2, column = 1, columnspan = 30, sticky = tk.SW + tk.NE)
        self.l_pickfile.grid(row = 4, column = 1, columnspan = 10, sticky = tk.S)
        self.b_pickfile.grid(row = 4, column = 11, columnspan = 5, sticky = tk.S + tk.E + tk.W)
        self.l_filecheck.grid(row = 4, column = 16, columnspan = 30, sticky = tk.S)
        self.l_setting.grid(row = 5, column = 1, columnspan = 30, sticky = tk.SW + tk.NE)
        self.l_remind.grid(row = 6, column = 1, columnspan = 30, sticky = tk.SW + tk.NE)

        self.l_Q1.grid(row = 7, column = 1, columnspan = 10, sticky = tk.SW)
        self.Q1_1.grid(row = 7, column = 11, columnspan = 7, sticky = tk.SW)
        self.Q1_2.grid(row = 7, column = 18, columnspan = 7, sticky = tk.SW)
        self.Q1_check.grid(row = 7, column = 25, columnspan = 6, sticky = tk.W + tk.E)

        self.l_Q2.grid(row = 8, rowspan = 2, column = 1, columnspan = 10, sticky = tk.W)
        self.Q2_1.grid(row = 8, column = 11, columnspan = 7, sticky = tk.SW)
        self.Q2_2.grid(row = 8, column = 18, columnspan = 7, sticky = tk.SW)
        self.Q2_3.grid(row = 9, column = 11, columnspan = 7, sticky = tk.SW)
        self.Q2_4.grid(row = 9, column = 18, columnspan = 7, sticky = tk.SW)
        self.Q2_check.grid(row = 8, rowspan = 2, column = 25, columnspan = 6, sticky = tk.W + tk.E)

        self.l_Q3.grid(row = 10, column = 1, columnspan = 10, sticky = tk.SW)
        self.Q3_1.grid(row = 10, column = 11, columnspan = 7, sticky = tk.SW)
        self.Q3_2.grid(row = 10, column = 18, columnspan = 7, sticky = tk.SW)

        self.b_continue.grid(row = 15, column = 1, columnspan = 3, sticky = tk.SW + tk.NE)

    #讓使用者找檔案
    def readfile(self):
        Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
        filename = askopenfilename(filetypes = [("txt files", "*.txt"), ("excel files", "*.csv"), ("excel files", "*.xlsx")])
        return filename 
        #file_name是路徑，之後需要可以叫它

    #事件處理函數，當點下「上傳檔案後」，呼叫找檔案的函數，並將文字改成檔案名稱
    def clickb_p(self):
        file_name = self.readfile()
        self.file_name = str(file_name)
        if file_name != '()':
            self.l_filecheck.configure(text = file_name)


    #檢視離群值
    def check_outlier(self):
        dataset=pp.dataset(file=self.file_name)
        isna=dataset.isna().sum()
        isna_str=str(isna)
        textlist=self.sp(isna_str)
        txtname=[]
        txtnum=[]
        for i in range(int(len(textlist)/2)):
            txtname.append(textlist[2*i]+'\n')
        for i in range(int(len(textlist)/2)):
            txtnum.append(textlist[2*i+1]+'\n')
        #
        outlier=tk.Tk()
        outlier.title('outlier')
        outlier.geometry('400x300')
        #
        wrapper1=tk.LabelFrame(outlier)
        wrapper1.grid(row=5,rowspan=10,column=0,columnspan=10)
        wrapper2=tk.LabelFrame(outlier)
        wrapper2.grid(row=0,rowspan=1,column=0,columnspan=10)
        #,width=500,height=600,sticky=tk.NE+tk.SW
        view_outlier=tk.Canvas(wrapper1,bg='PowderBlue')
        view_outlier.grid(row=10,column=0,columnspan=15,sticky=tk.NE+tk.SW)
        #
        gscrollbar=ttk.Scrollbar(wrapper1,orient='vertical',command=view_outlier.yview)
        gscrollbar.grid(row=10,column=15,sticky=tk.S + tk.E + tk.N)
        view_outlier.configure(yscrollcommand=gscrollbar.set)
        view_outlier.bind('<Configure>',lambda e:view_outlier.configure(scrollregion=view_outlier.bbox('all')))
        #
        myframe=tk.Frame(view_outlier,bg='PowderBlue')
        view_outlier.create_window((50,10),window=myframe,anchor='nw')
        for i in range(len(txtnum)):
            tk.Label(myframe,text=txtname[i],bg='PowderBlue').grid(row=i+1,column=0,columnspan=3,sticky=tk.NW)
            tk.Label(myframe,text=txtnum[i],bg='PowderBlue').grid(row=i+1,column=2,columnspan=3,sticky=tk.NE)
        outlier.mainloop()
    #def strtoword(self,inputstr):
    def sp(self,aa):
        aaa=[]
        aa=aa.split('  ')
        for i in range(len(aa)):
            aa[i]=aa[i].split('\n')
        aa=[x for i in aa for x in i]
        for i in range(len(aa)):
            if aa[i]!="":
                aaa.append(aa[i])
        return aaa
        
        

    #檢視缺失值
    #def check_missing(self):

    #點下「下一步」時，確認是否已上傳檔案、完成填寫，如有則跳下一部分；未達成則提醒
    def click_con(self):
        fliecheck = self.l_filecheck.cget('text')
        if fliecheck == '您尚未選擇檔案':
            messagebox.showerror('error', '您尚未選擇檔案')
        else:
            run_tree = tk.Tk()
            run_tree.title('Result')

            #點擊「計算模型效能」，要產生東西的函數～
            def click_perform():
                dataset = pp.dataset(file=self.file_name)
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

                c_performance.create_text(400, 120, text=out, fill="black",font="Times 20 italic bold")


            #點擊「生成決策樹」，要產生樹的函數～
            def click_tree():
                imagemain=ImageTk.PhotoImage(file='treewenwen.png')
                self.c_tree.create_image(0,0,image=imagemain,anchor=tk.NW)
                
    
            b_performance = tk.Button(run_tree, text = '模型計算效能', bg = 'LightCoral', font = ('Arial', 32),command=lambda:click_perform)
            b_tree = tk.Button(run_tree, text = '生成決策樹', bg = 'LightCoral', font = ('Arial', 32),command=click_tree)  #,command=click_tree
            c_performance = tk.Canvas(run_tree, width = 800, height = 200, bg = 'PowderBlue')
            self.c_tree = tk.Canvas(run_tree, width = 800, height = 400, bg = 'PowderBlue')

            b_performance.grid(row = 1, rowspan = 2, column = 2, sticky = tk.NE + tk.SW)
            c_performance.grid(row = 3, column = 2, sticky = tk.NE + tk.SW)
            b_tree.grid(row = 4, rowspan = 2, column = 2, sticky = tk.NE + tk.SW)
            self.c_tree.grid(row = 6, column = 2, sticky = tk.NE + tk.SW)
            

            run_tree.mainloop()

tree = maketree()
tree.master.title('The Decision Tree Tool')
tree.mainloop()
