import tkinter as tk
import tkinter.font as tkFont
from tkinter import Tk, messagebox, Canvas, ttk
from tkinter.filedialog import askopenfilename
import PIL
from PIL import Image, ImageTk
import pydotplus
from pydotplus import graph_from_dot_data
import os
import pylab
from IPython.display import display
import numpy as np
from sklearn import tree
from sklearn.impute import KNNImputer
import pandas as pd

import DecisionTree_Mushroom as DM
import PBC80.preprocessor as pp
from PBC80.preprocessor import KBestSelector
from PBC80.classification import DecisionTree
from PBC80.performance import KFoldClassificationPerformance
import PBC80.model_drawer as md

class maketree(tk.Frame):
    def __init__(self):
        tk.Frame.__init__(self)
        self.grid()
        self.createWidgets()

    #--------------------------------------創造不同欄位-------------------------#
    def createWidgets(self):
        f1 = tkFont.Font(size = 16, family = 'Arial')
        f2 = tkFont.Font(size = 20, family = 'Arial')
        f3 = tkFont.Font(size = 28, family = 'Arial')
        f4 = tkFont.Font(size = 16, family = 'Arial')

        radioValue = tk.IntVar()
        radioValue2 = tk.IntVar()
        radioValue3 = tk.IntVar()
        
        intro = '歡迎使用決策樹產生器！\n請選擇需進行資料整理的檔案，並依序設定是否進行離群值剔除、補齊缺失值、降維度後，按下開始鍵，即可將資料進行分群並生成決策樹'
        self.l_title = tk.Label( self, text = '決策樹產生器', bg = 'PowderBlue', fg = 'CadetBlue', font = f3)
        self.m_intro = tk.Message( self, text = intro, bg = 'Gainsboro', aspect = 1000, font = f1)
        self.l_pickfile = tk.Label( self, text = '請上傳檔案：', fg = 'black', font = f2)
        self.b_pickfile = tk.Button( self, text = '選擇檔案', bg = 'Gainsboro', fg = 'gray', command = self.clickb_p, font = f2)
        self.l_filecheck = tk.Label( self, text = '您尚未選擇檔案',anchor=tk.SW, wraplength=500,fg = 'gray',justify='left',height=3 ,font = f1)
        self.l_setting = tk.Label(self, text = '資料處理設定', bg = 'LightCoral', fg = 'black', font = f2)
        
        remind = '* 如未進行設定，則預設為：剔除離群值、以KNN Imputer方式補齊缺失值、進行降維處理。'
        self.l_remind = tk.Label( self, text = remind, bg = 'LightCoral', fg = 'Snow', font = f1)
        
    #------------------------第一題：總覽
        self.l_Q1 = tk.Label(self, text = '1. 點擊以總覽資料 !', font = f4)
        self.Q1_check = tk.Button(self, text = '資料總覽', bg = 'gray', fg = 'black',command=self.check_information_f, font = f4)
        s=ttk.Style()
        s.configure('red.TSeparator',background='red')
        b=ttk.Separator(self,orient='horizontal',style='red.TSeparator').grid(row =9, column = 1, columnspan = 30, sticky = tk.SW + tk.NE)
    #------------------------第二題：離群值
        self.l_Q2 = tk.Label(self, text = '2. 是否要踢除離群值(outlier)？', font = f4)
        self.Q2_1 = tk.Radiobutton(self, text = '是，以缺失值處理', variable = radioValue, value = 1 , font = f4)
        self.Q2_2 = tk.Radiobutton(self, text = '否，保留離群值', variable = radioValue, value = 2, font = f4)
        self.Q2_check = tk.Button(self, text = '檢視outlier比率', bg = 'gray', fg = 'black', font = f4,command=self.check_outlier)
        s=ttk.Style()
        s.configure('red.TSeparator',background='red')
        b=ttk.Separator(self,orient='horizontal',style='red.TSeparator').grid(row =13, column = 1, columnspan = 30, sticky = tk.SW + tk.NE)   
    #------------------------第三題：缺失值
        self.knnboo=False
        self.l_Q3 = tk.Label(self, text = '3. 選擇缺失值替補方式 !', font = f4)
        self.Q3_1 = tk.Radiobutton(self, text = '無', variable = radioValue2, value = 1 , font = f4,command=lambda:self.knncheck(0))
        self.Q3_2 = tk.Radiobutton(self, text = '中位數', variable = radioValue2, value = 2, font = f4)
        self.Q3_3 = tk.Radiobutton(self, text = '眾數', variable = radioValue2, value = 3 , font = f4)
        self.Q3_4 = tk.Radiobutton(self, text = 'KNN Imputer', variable = radioValue2, value = 4, font = f4,command=lambda:self.knncheck(1))
        self.Q3_check = tk.Button(self, text = '檢視缺失值數量', bg = 'gray', fg = 'black', font = f4,command=lambda:self.check_missing(self.knnboo))
        s=ttk.Style()
        s.configure('red.TSeparator',background='red')
        b=ttk.Separator(self,orient='horizontal',style='red.TSeparator').grid(row =16, column = 1, columnspan = 30, sticky = tk.SW + tk.NE)   
    #------------------------第四題：降維度
        self.l_Q4 = tk.Label(self, text = '4. 是否要將資料降維處理(PCA)？', font = f4)
        self.Q4_1 = tk.Radiobutton(self, text = '是', variable = radioValue3, value = 1 , font = f4)
        self.Q4_2 = tk.Radiobutton(self, text = '否', variable = radioValue3, value = 2, font = f4)
      
     #-----------------------輸完資料後開始進行處理
        self.b_continue = tk.Button(self, text = '下一步', command = self.click_con, font = f2)
        self.l_title.grid(row = 1, column = 1, columnspan = 30, sticky = tk.SW + tk.NE)
        self.m_intro.grid(row = 2, column = 1, columnspan = 30, sticky = tk.SW + tk.NE)
        self.l_pickfile.grid(row = 5, column = 1, columnspan = 5, sticky = tk.S)
        self.b_pickfile.grid(row = 5, column = 6, columnspan = 5, sticky = tk.S + tk.E + tk.W)
        self.l_filecheck.grid(row = 5, column = 11, columnspan =10, sticky = tk.NE+tk.SW)
        self.l_setting.grid(row = 6, column = 1, columnspan = 30, sticky = tk.SW + tk.NE)
        
        
        self.l_Q1.grid(row = 8, column = 1, columnspan = 10, sticky = tk.SW)
        self.Q1_check.grid(row = 8, column = 25, columnspan = 6, sticky = tk.W )
        self.l_Q2.grid(row = 11, rowspan=2,column = 1, columnspan = 10, sticky = tk.W)
        self.Q2_1.grid(row = 11, column = 11, columnspan = 7, sticky = tk.SW)
        self.Q2_2.grid(row = 12, column = 11, columnspan = 7, sticky = tk.SW)
        self.Q2_check.grid(row = 11,rowspan=2, column = 25, columnspan = 6, sticky = tk.W + tk.E)

        self.l_Q3.grid(row = 14, rowspan = 2, column = 1, columnspan = 10, sticky = tk.W)
        self.Q3_1.grid(row = 14, column = 11, columnspan = 7, sticky = tk.SW)
        self.Q3_2.grid(row = 14, column = 18, columnspan = 7, sticky = tk.SW)
        self.Q3_3.grid(row = 15, column = 11, columnspan = 7, sticky = tk.SW)
        self.Q3_4.grid(row = 15, column = 18, columnspan = 7, sticky = tk.SW)
        self.Q3_check.grid(row = 14, rowspan = 2, column = 25, columnspan = 6, sticky = tk.W + tk.E)

        self.l_Q4.grid(row = 17, column = 1, columnspan = 10, sticky = tk.SW)
        self.Q4_1.grid(row = 17, column = 11, columnspan = 7, sticky = tk.SW)
        self.Q4_2.grid(row = 17, column = 18, columnspan = 7, sticky = tk.SW)
        #self.Q4_check.grid(row = 17, rowspan = 2, column = 25, columnspan = 6, sticky = tk.W + tk.E)

        self.b_continue.grid(row = 20, column = 1, columnspan = 3, sticky = tk.SW + tk.NE)
  
     
    #-------------------------------讓使用者找檔案------------------------------#
    def readfile(self):
        root = Tk()
        root.withdraw()# we don't want a full GUI, so keep the root window from appearing
        filename = askopenfilename(filetypes = [("excel files", "*.csv"), ("excel files", "*.xlsx"), ("txt files", "*.txt")])
        root.destroy()
        return filename
        #file_name是路徑，之後需要可以叫它

    #事件處理函數，當點下「上傳檔案後」，呼叫找檔案的函數，並將文字改成檔案名稱
    def clickb_p(self):
        file_name = self.readfile()
        file_name = str(file_name)
        if file_name != '()':
            self.l_filecheck.configure(text = file_name)
    #-------------------------------檢視information-----------------------------#         
    def is_number(self,s):
        try:
            float(s)
            return True
        except ValueError:
            pass
        try:
            import unicodedata
            unicodedata.numeric(s)
            return True
        except (TypeError,ValueError):
            pass 
    def check_information_f(self):    
        ##
        File = self.l_filecheck.cget('text')
        dataset = pp.dataset(file = File)    
        data_discribe=dataset.describe()
        data_discrib=data_discribe.rename_axis('index').reset_index()
        missing=tk.Toplevel()
        missing.title('check_information')
        missing.geometry('400x300')
        #
        wrapper1=tk.LabelFrame(missing)
        wrapper1.grid(row=5,rowspan=10,column=0,columnspan=10)
        #
        view_outlier=tk.Canvas(wrapper1,bg='PowderBlue')
        view_outlier.grid(row=10,column=0,columnspan=15,sticky=tk.NE+tk.SW)
        #--------------------------滾輪
        '''
        沒有bind_all，可以控制的區域就只有canvas不包含myframe的部分
        '''
        gscrollbar=ttk.Scrollbar(wrapper1,orient='horizontal',command=view_outlier.xview)
        gscrollbar.grid(row=20,column=0,columnspan=15,sticky=tk.W +tk.E)
        view_outlier.configure(xscrollcommand=gscrollbar.set)
        view_outlier.bind_all("<MouseWheel>", lambda e:view_outlier.xview("scroll",int(-1*(e.delta/23)),"units"))
        view_outlier.bind('<Configure>',lambda e:view_outlier.configure(scrollregion=view_outlier.bbox('all')))
        #
        myframe=tk.Frame(view_outlier,bg='PowderBlue')
        view_outlier.create_window((50,10),window=myframe,anchor='nw')
        #
        co=data_discrib.columns
        my_tree=ttk.Treeview(myframe,show='headings',columns=co)
        style = ttk.Style()
        style.theme_use('default')
        style.configure("Treeview.Heading",foreground='#005AB5',background='#ACD6FF', font=('Comic Sans MS', 8))
        style.map('Treeview',background=[('selected','#AE57A4')])
        for i in range(len(co)):
            my_tree.column(i,width=110)
            my_tree.heading(i,text=co[i])
            
        my_tree.grid(row=8,column=0)
        temarr=np.array(data_discrib) 
        for i in range(len(temarr)):
            my_tree.insert('',i,values=temarr[i])
        missing.mainloop()

                
    #---------------------------------檢視離群值--------------------------------#
    def check_outlier(self):
        flag = 0
        out = tk.Toplevel()
        out.title('Outlier')
        File = self.l_filecheck.cget('text')
        dataset = pp.dataset(file = File)
        X = pp.decomposition(dataset, x_columns=[i for i in range(1, 18)])
        for i, column in enumerate(X.columns):
            data = X[column]
            a = self.outlier_percent(data)
            if (a == -1):
                flag = 1
                break
            percent = str(round(a, 2))
            a = tk.Label(out, text = f'Outliers in "{column}": {percent}%', font = tkFont.Font(size = 16, family = 'Arial'))
            a.grid(row = i + 1, columnspan = 30, sticky = tk.W)
            #print(f'Outliers in "{column}": {percent}%')
        if flag:
            out.destroy()
            messagebox.showerror('error', '資料型態非數字，無法計算離群值，請在選項選擇「否」。')
        else:
            out.mainloop()

    def outlier_percent(self, data):
        try:
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            minimum = Q1 - (1.5 * IQR)
            maximum = Q3 + (1.5 * IQR)
            num_outliers =  np.sum((data < minimum) |(data > maximum))
            num_total = data.count()
            return (num_outliers/num_total)*100
        except TypeError:
            return -1
        
    #---------------------------------檢視缺失值--------------------------------#  
    def knncheck(self,x):
        if x==0:
            self.knnboo=False
        else :
            self.knnboo=True
    def check_missing(self,boo):
        def knn_check_missing(dataframe):
            for i in dataframe:
                if  not self.is_number(dataframe[i].iat[0]):
                    count=0
                    dic={}
                    for j in dataframe[i]:
                        if j not in dic and j!='' :
                            dic[j]=count
                            count+=1
                    for j in range(len(dataframe[i])):
                        dataframe[i].iat[j]=dic[dataframe[i].iat[j]]
        
            inputer=KNNImputer(n_neighbors=2)
            aa=np.array(inputer.fit_transform(dataframe.values))
            da={}
            count=0
            for i in dataframe:
                da[i]={}
                for j in range(len(dataframe[i])):
                    da[i][j]=aa[j][count]
                count+=1
            return da
        
        File = self.l_filecheck.cget('text')
        dataset = pp.dataset(file = File)
        if boo:
            dataset = pp.dataset(file = File)
            self.knn_dataset=pd.DataFrame(knn_check_missing(dataset))
        else:
            self.knn_dataset=pp.dataset(file=File)
        isna=self.knn_dataset.isna().sum()
        isna_str=str(isna)
        textlist=self.sp(isna_str)
        txtname=[]
        txtnum=[]
        for i in range(int(len(textlist)/2)):
            txtname.append(textlist[2*i]+'\n')
        for i in range(int(len(textlist)/2)):
            txtnum.append(textlist[2*i+1]+'\n')
        #
        missing=tk.Tk()
        missing.title('missing')
        missing.geometry('400x300')
        #
        wrapper1=tk.LabelFrame(missing)
        wrapper1.grid(row=5,rowspan=10,column=0,columnspan=10)
        wrapper2=tk.LabelFrame(missing)
        wrapper2.grid(row=0,rowspan=1,column=0,columnspan=10)
        #
        view_outlier=tk.Canvas(wrapper1,bg='PowderBlue')
        view_outlier.grid(row=10,column=0,columnspan=15,sticky=tk.NE+tk.SW)
        #
        gscrollbar=ttk.Scrollbar(wrapper1,orient='vertical',command=view_outlier.yview)
        gscrollbar.grid(row=10,column=15,sticky=tk.S + tk.E + tk.N)
        view_outlier.configure(yscrollcommand=gscrollbar.set)
        view_outlier.bind_all("<MouseWheel>", lambda e:view_outlier.yview("scroll",int(-1*(e.delta/30)),"units"))
        view_outlier.bind('<Configure>',lambda e:view_outlier.configure(scrollregion=view_outlier.bbox('all')))
        #
        myframe=tk.Frame(view_outlier,bg='PowderBlue')
        view_outlier.create_window((50,10),window=myframe,anchor='nw')
        for i in range(len(txtnum)):
            tk.Label(myframe,text=txtname[i],bg='PowderBlue').grid(row=i+1,column=0,columnspan=3,sticky=tk.NW)
            tk.Label(myframe,text=txtnum[i],bg='PowderBlue').grid(row=i+1,column=2,columnspan=3,sticky=tk.NE)
        missing.mainloop()
        
        
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
    #----------------------------------pca---------------------------------------#
    def pca_click(self):
        print('yes1')
        pca_c=pp.PCASelector('auto')
        x_columns=np.array(self.knn_dataset.columns)
        x_columns = x_columns[1:]; x_columns
        X, Y = pp.decomposition(self.knn_dataset, x_columns, y_columns=[0])
        x_verbose=pca_c.fit(X,True,True)
        ny=x_verbose[1]
        #
        pca_cli=tk.Tk()
        pca_cli.title('missing')
        pca_cli.geometry('400x300')
        #
        wrapper1=tk.LabelFrame(pca_cli)
        wrapper1.grid(row=5,rowspan=10,column=0,columnspan=10)
        wrapper2=tk.LabelFrame(pca_cli)
        wrapper2.grid(row=0,rowspan=1,column=0,columnspan=10)
        #
        view_outlier=tk.Canvas(wrapper1,bg='PowderBlue')
        view_outlier.grid(row=10,column=0,columnspan=15,sticky=tk.NE+tk.SW)
        #
        gscrollbar=ttk.Scrollbar(wrapper1,orient='vertical',command=view_outlier.yview)
        gscrollbar.grid(row=10,column=15,sticky=tk.S + tk.E + tk.N)
        view_outlier.configure(yscrollcommand=gscrollbar.set)
        view_outlier.bind_all("<MouseWheel>", lambda e:view_outlier.yview("scroll",int(-1*(e.delta/30)),"units"))
        view_outlier.bind('<Configure>',lambda e:view_outlier.configure(scrollregion=view_outlier.bbox('all')))
        #
        myframe=tk.Frame(view_outlier,bg='PowderBlue')
        view_outlier.create_window((50,10),window=myframe,anchor='nw')
        tk.Label(myframe,text='pca_dimmation_reduce',bg='PowderBlue').grid(row=0,column=0,columnspan=3,sticky=tk.NW)
        co=['cumulat','convergence of information']
        my_tree=ttk.Treeview(myframe,show='headings',columns=co)
        style1 = ttk.Style()
        style1.theme_use('default')
        style1.configure("Treeview.Heading",foreground='#005AB5',background='#ACD6FF', font=('Comic Sans MS', 8))
        style1.map('Treeview',background=[('selected','#AE57A4')])
        for i in range(len(co)):
            my_tree.column(i,width=110)
            my_tree.heading(i,text=co[i])
        my_tree.grid(row=10,column=0)
        temarr=np.array(ny) 
        for i in range(len(temarr)):
            a=temarr[i][0]
            b=temarr[i][1]
            my_tree.insert('',i,values=(a,b))
        pca_cli.mainloop()
        



    #-點下「下一步」時，確認是否已上傳檔案、完成填寫，如有則跳下一部分；未達成則提醒-#
    def click_con(self):
        filecheck = self.l_filecheck.cget('text')
        if filecheck == '您尚未選擇檔案':
            messagebox.showerror('error', '您尚未選擇檔案')
        else:
            run_tree = tk.Toplevel()
            run_tree.title('Result')

            #點擊「計算模型效能」，要產生東西的函數～
            #def click_perform(run_tree):

            #點擊「生成決策樹」，要產生樹的函數～
            #def click_tree(run_tree):
            b_performance = tk.Button(run_tree, text = '模型計算效能', bg = 'LightCoral', command = self.clickBtnLo, font = ('Arial', 32))
            b_tree = tk.Button(run_tree, text = '生成決策樹', bg = 'LightCoral', command = self.click_tree, font = ('Arial', 32))
            self.c_performance = tk.Canvas(run_tree, width = 800, height = 200, bg = 'PowderBlue')
            #self.c_tree = tk.Canvas(run_tree, width = 800, height = 400, bg = 'PowderBlue')

            b_performance.grid(row = 1, rowspan = 2, column = 2, sticky = tk.NE + tk.SW)
            self.c_performance.grid(row = 3, column = 2, sticky = tk.NE + tk.SW)
            b_tree.grid(row = 4, rowspan = 2, column = 2, sticky = tk.NE + tk.SW)
            #self.c_tree.grid(row = 6, column = 2, sticky = tk.NE + tk.SW)
            run_tree.mainloop()

    def clickBtnLo(self):
        f = self.l_filecheck.cget('text')
        self.runmodel(f)

    def click_tree(self):
        photo = tk.Toplevel()
        File = self.l_filecheck.cget('text')
        DM.draw_picture(File)
        img = Image.open('decision_tree_graphivz.png')
        img2 = ImageTk.PhotoImage(img)

        myimage = tk.Canvas(photo, width=img.size[0], height=img.size[1])
        myimage.pack()
        myimage.create_image(0,0, anchor=tk.NW, image=img2)
        photo.mainloop()

    def runmodel(self,f):
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
        self.c_performance.create_text(400, 120, text=out, fill="black",font="Times 20 italic bold")


tree1 = maketree()
tree1.master.title('The Decision Tree Tool')
tree1.mainloop()
