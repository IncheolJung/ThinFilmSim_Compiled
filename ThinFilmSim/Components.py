import numpy as np
import pandas as pd
import tkinter as tk
import tkinter.ttk as ttk
import os, winsound, csv, time, copy
import matplotlib.pyplot as plt
import matplotlib.patches as pch
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg,NavigationToolbar2Tk
from scipy.interpolate import interp1d
from scipy.optimize import minimize



class AutocompleteEntry(ttk.Combobox):

        def set_completion_list(self, completion_list):
                """Use our completion list as our drop down selection menu, arrows move through menu."""
                self._completion_list = sorted(completion_list, key=str.lower) # Work with a sorted list
                self._hits = []
                self._hit_index = 0
                self.position = 0
                self.bind('<KeyRelease>', self.handle_keyrelease)
                self['values'] = self._completion_list  # Setup our popup menu

        def autocomplete(self, delta=0):
                """autocomplete the Combobox, delta may be 0/1/-1 to cycle through possible hits"""
                if delta: # need to delete selection otherwise we would fix the current position
                        self.delete(self.position,"end")
                else: # set position to end so selection starts where textentry ended
                        self.position = len(self.get())
                # collect hits
                _hits = []
                for element in self._completion_list:
                        if element.lower().startswith(self.get().lower()): # Match case insensitively
                                _hits.append(element)
                # if we have a new hit list, keep this in mind
                if _hits != self._hits:
                        self._hit_index = 0
                        self._hits=_hits
                # only allow cycling if we are in a known hit list
                if _hits == self._hits and self._hits:
                        self._hit_index = (self._hit_index + delta) % len(self._hits)
                # now finally perform the auto completion
                if self._hits:
                        self.delete(0,"end")
                        self.insert(0,self._hits[self._hit_index])
                        self.select_range(self.position,"end")

        def handle_keyrelease(self, event):
                """event handler for the keyrelease event on this widget"""
                if event.keysym == "BackSpace":
                        self.delete(self.index(tk.INSERT),"end")
                        self.position = self.index("end")
                if event.keysym == "Left":
                        if self.position < self.index("end"): # delete the selection
                                self.delete(self.position,"end")
                        else:
                                self.position = self.position-1 # delete one character
                                self.delete(self.position,"end")
                if event.keysym == "Right":
                        self.position = self.index("end") # go to end (no selection)
                if len(event.keysym) == 1:
                        self.autocomplete()
                # No need for up/down, we'll jump to the popup
                # list at the position of the autocompletion

class AutoCombobox(ttk.Combobox):
    def __init__(self, parent, **options):
        ttk.Combobox.__init__(self, parent, **options)
        self.bind("<KeyRelease>", self.AutoComplete_1)
        self.bind("<<ComboboxSelected>>", self.Cancel_Autocomplete)
        self.bind("<Return>", self.Cancel_Autocomplete)
        self.bind("<Down>",None)
        self.autoid = None

    def Cancel_Autocomplete(self, event=None):
        self.after_cancel(self.autoid) 

    def AutoComplete_1(self, event):
        if self.autoid != None:
            self.after_cancel(self.autoid)
        if event.keysym in ["BackSpace", "Delete", "Return"]:
            return
        self.autoid = self.after(200, self.AutoComplete_2)

    def AutoComplete_2(self):
        data = self.get()
        if data != "":
            for entry in self["values"]:
                match = True
                try:
                    for index in range(0, len(data)):
                        if data[index] != entry[index]:
                            match = False
                            break
                except IndexError:
                    match = False
                if match == True:
                    self.set(entry)
                    self.selection_range(len(data), "end")
                    self.event_generate("<Down>",when="tail")
                    self.focus_set()
                    break
            self.autoid = None

class StructureTable():
    
    def __init__(self,masterApp,frame,structure=[],
                 columns=["Material","Thickness","FitRange","DepoAngle","FitRange","Coherence",],
                 font=("calibri",12,""),refdata=[],showmediums=True):
        self.App = masterApp
        self.mainframe = tk.Frame(frame,relief="solid",bd=1.5)
        self.structure = structure
        self.length = len(structure)
        columns.insert(0,"index")
        self.cols = columns
        self.font,self.fontsize,self.fontweight = font
        self.materials = sorted(refdata.keys())
        self.materials_lower = [e.lower() for e in self.materials]
        self.showmediums = showmediums
        self.selection = [0,0]
        self.oldstructures = [self.structure]
        self.newstructures = []
        self.filepath = "Save/"
        self.createUI()
        return
    
    def createUI(self):

        def Scrollregion(event=None):
            canvas.configure(scrollregion=canvas.bbox("all"))
            return
        
        def OnMouseWheel(event=None):
            ii,jj = self.selection
            if len(self.structure) > 11:
                if jj != 1: canvas.yview_scroll(-1*int(event.delta/120), "units")
            return

        for e in self.mainframe.pack_slaves(): e.destroy()
        canvas = tk.Canvas(self.mainframe,width=700,height=350,
                           highlightthickness=0)
        scrollbar = tk.Scrollbar(self.mainframe,orient="vertical",
                                 command=canvas.yview)
        canvas.pack(side="left",expand=True,)
        scrollbar.pack(side="right",fill="y",)
        structureframe = tk.Frame(canvas)
        structureframe.bind("<Configure>",Scrollregion)
        structureframe.bind_all("<MouseWheel>",OnMouseWheel)
        canvas.create_window((0,0),window=structureframe,anchor="nw",)
        canvas.configure(yscrollcommand=scrollbar.set)

        self.entries = []
        structure = copy.deepcopy(self.structure)
        length = len(structure)-1
        for jj,col in enumerate(self.cols):
            e = tk.Label(structureframe,text=col,font=(self.font,self.fontsize,"bold"))
            e.grid(row=0,column=jj)
        for ii,row in enumerate(structure):
            if self.showmediums:
                if ii==0: row.insert(0,"incident")
                elif ii==length: row.insert(0,"exit")
                else: row.insert(0,ii)
            else: row.insert(0,ii)
            entries_row = []
            for jj,col in enumerate(row):
                if jj==0:
                    string = tk.StringVar()
                    e = tk.Entry(structureframe,width=6,justify="center",
                                       textvariable=string,state="readonly",
                                       font=(self.font,self.fontsize,""),)
                    string.set(str(col))
                elif jj==1:
                    idx_tkn = True
                    try: index = self.materials.index(col)
                    except ValueError:
                        if type(col) is list:
                            tokens = [True]
                        elif type(col) is str:
                            if col[0] == "[":  col = col[1:]
                            if col[-1] == "]": col = col[:-1]
                            col_ = [e.lstrip().rstrip().split("%")[0] for e in col.split(";")]
                            chk_lst = ["l","linear","b","bruggeman","g","glad",]
                            if col_[0].lower() in chk_lst:
                                col_ = col_[1:]
                            tokens = [c in self.materials for c in col_]
                        elif type(col) is dict:
                            col_ = col["material"]
                            tokens = [c in self.materials for c in col_]
                        else: tokens = [False]
                        if all(tokens):
                            idx_tkn = False
                            material_name = str(col)
                        else:
                            index = 0
                            self.App.app_info.set("Warning: Material Not Found")
                            self.App.BEEP("fail")
                    e = ttk.Combobox(structureframe,width=10,height=10,
                                     values=self.materials,
                                     font=(self.font,self.fontsize,""),)
                    e.unbind_class("TCombobox","<Down>")
                    e.unbind_class("TCombobox","<MouseWheel>")
                    if idx_tkn: e.current(index)
                    else: e.insert(0,material_name)
                else:
                    string = tk.StringVar()
                    e = tk.Entry(structureframe,width=10,
                                   textvariable=string,justify="center",
                                   font=(self.font,self.fontsize,""),)
                    string.set(str(col))
                e.grid(row=ii+1,column=jj)
                e.bind("<Key>",lambda event: self.shortcuts(event))
                e.bind("<Control-Key>",lambda event: self.shortcuts(event,"Control"))
                e.bind("<Alt-Key>",lambda event: self.shortcuts(event,"Alt"))
                e.bind("<Control-Shift-Key>",lambda event: self.shortcuts(event,"Control-Shift"))
                e.bind("<Button-1>",self.shortcuts)
                entries_row.append(e)
            self.entries.append(entries_row)
        self.length = ii+1
        return
    
    def update(self,structure=[],refdata=[],redo_flg=False,undo_flg=False,forced_flg=False):
        self.App.app_info.set("Creating Structure....")
        if structure: self.structure = copy.deepcopy(structure)
        # else:         self.structure = self.get()
        if refdata:
            self.materials = sorted(refdata.keys())
            self.materials_lower = [e.lower() for e in self.materials]
        if undo_flg:
            self.newstructures.append(self.get())
        else:
            if not redo_flg: self.newstructures = []
            curstructure = self.get()
            if not self.structure==curstructure:
                self.oldstructures.append(curstructure)
        if len(self.structure)==self.length and not forced_flg:
            for layer,entries_row in zip(self.structure,self.entries):
                temp = entries_row.pop(0)
                for col,e in zip(layer,entries_row):
                    e.delete(0,"end")
                    e.insert(0,str(col))
                entries_row.insert(0,temp)
        else: self.createUI()
        ii,jj = self.selection
        try: self.entries[ii][jj].focus_set()
        except:
            ii,jj = 0,1
            self.selection = [ii,jj]
            self.entries[ii][jj].focus_set()
        self.App.Selectors.update()
        self.App.app_info.set("Now Ready")
        return
    
    def undo(self):
        if self.oldstructures:
            self.structure = self.oldstructures.pop(-1)
            self.update(undo_flg=True)
        return
    
    def redo(self):
        if self.newstructures:
            self.structure = self.newstructures.pop(-1)
            self.update(redo_flg=True)
        return
    
    def new(self):
        self.structure = [
            ["Air",0,[],0,[],0,],
            ["TiO2",0,[0,None],0,[],1,],
            ["Glass",1e6,[],0,[],0,],
            ["Air",0,[],0,[],0,],
        ]
        self.App.structure = self.structure
        self.App.app_info.set("Now Ready")
        self.update()
        return
    
    def reverse(self):
        self.structure = list(reversed(self.structure))
        self.App.structure = self.structure
        self.update()
        return
    
    def move(self,state):
        ii,jj = self.selection
        if state == "down": newii = ii+1
        elif state == "up": newii = ii-1
        self.structure.insert(newii,self.structure.pop(ii))
        self.selection = newii,jj
        self.update()
        return
    
    def duplicate(self):
        ii,jj = self.selection
        newlayer = copy.deepcopy(self.structure[ii])
        self.structure.insert(ii,newlayer)
        self.update()
        return
    
    def add(self):
        ii,jj = self.selection
        newlayer = ["TiO2",0,[0,None],0,[],1,]
        self.structure.insert(ii,newlayer)
        self.update()
        return
    
    def delete(self):
        ii,jj = self.selection
        del self.structure[ii]
        self.update()
        return
    
    def makeCavity(self):
        ii,jj = self.selection
        material_name,coherence = "Cavity",0
        self.structure[ii][1] = material_name
        self.structure[ii][5] = coherence
        self.entries[ii][2].delete(0,"end")
        self.entries[ii][6].delete(0,"end")
        self.entries[ii][2].insert(0,str(material_name))
        self.entries[ii][6].insert(0,str(coherence))
        return
    
    def makePeriodic(self):
        ii,jj = self.selection
        font = (self.font,self.fontsize,self.fontweight)
        self.App.app_info.set("Creating Periodic Structure")
        config = PeriodicSettings(self.App,font=font,cur_layer=self.structure[ii])
        res = config.get()
        if res:
            self.delete()
            for layer in reversed(res):
                self.structure.insert(ii,layer)
        config.terminate()
        self.update()
        return

    def makeInducedTransmission(self):
        ii,jj = self.selection
        font = (self.font,self.fontsize,self.fontweight)
        self.App.app_info.set("Creating Induced Transmission Filter")
        self.App.wvl = float(self.App.wvlth.get())
        incident,exit = self.structure[ii-1][0],self.structure[ii+1][0]
        config = InducedTransmissionFilter(self.App,incident,exit,font=font,)
        res = config.get()
        if res:
            self.delete()
            for layer in reversed(res): self.structure.insert(ii,layer)
        config.terminate()
        self.update()
        return
    
    def makeQuadWave(self):
        ii,jj = self.selection
        material_name,coherence = "Quad Wavelength Layer",1
        self.structure[ii][1] = material_name
        self.structure[ii][5] = coherence
        self.entries[ii][2].delete(0,"end")
        self.entries[ii][6].delete(0,"end")
        self.entries[ii][2].insert(0,str(material_name))
        self.entries[ii][6].insert(0,str(coherence))
        return
    
    def makeVariable(self):
        ii,jj = self.selection
        if jj < 3: jj = 1
        else:      jj = 3
        material_name = "Variable"
        self.structure[ii][jj] = material_name
        self.entries[ii][jj+1].delete(0,"end")
        self.entries[ii][jj+1].insert(0,str(material_name))
        try:
            lb,ub = self.structure[ii][jj+1]
            if lb is None or ub is None: raise ValueError
        except ValueError:
            fitrng = f"[{min(self.App.angles)},{max(self.App.angles)}]"
            self.entries[ii][jj+2].delete(0,"end")
            self.entries[ii][jj+2].insert(0,str(fitrng))
        return
    
    def load(self,):
        filename = tk.filedialog.askopenfilename(initialdir=self.filepath,title="select a file",
                            filetypes =(("dat files","*.dat"),("all files","*.*")))
        try:
            structure = []
            with open(filename,"r") as f:
                data = f.read().splitlines()
                for row in data:
                    data_row = eval(row)
                    structure.append(data_row)
            
            if len(structure[0])==4:
                self.App.app_info.set("Updating Structure Data....")
                for ii,layer in enumerate(structure):
                    mat,thi,coh,FRg = layer
                    structure[ii] = [mat,thi,FRg,0,[],coh,]
                with open(filename,"w") as f:
                    data = structure
                    for row in data: f.write(str(row)+"\n")

            self.structure = structure
            self.App.structure = self.structure
            self.App.app_info.set("Structure Loaded")
            self.update()
        except: self.App.app_info.set("Failed to Load Structure")
        self.filepath = ""
        for e in filename.split("/")[:-1]: self.filepath = self.filepath+"/"+e
        return
    
    def save(self,):
        self.structure = self.get()
        filename = tk.filedialog.asksaveasfilename(initialdir=self.filepath,title="select a file",
                            filetypes =(("dat files","*.dat"),("all files","*.*")))
        try:
            if filename.split(".")[0]==filename: filename = filename+".dat"
            with open(filename,"w") as f:
                data = self.structure
                for row in data: f.write(str(row)+"\n")
            self.filepath = filename
            self.App.app_info.set("Structure Saved")
        except: self.App.app_info.set("Failed to Save Structure")
        return
    
    def pack(self,*args):
        self.mainframe.pack(*args)
        return
    
    def focus_set(self,*args):
        ii,jj = self.selection
        if not jj: jj+=1
        selectedWidget = self.entries[ii][jj]
        try: selectedWidget.focus_set()
        except: print("Not available")
        selectedWidget.select_range(0,"end")
        return
    
    def get(self):
        values = []
        for row in self.entries:
            values_row = []
            for ii,e in enumerate(row):
                if ii:
                    try: values_row.append(eval(e.get()))
                    except (NameError,SyntaxError): values_row.append(e.get())
            values.append(values_row)
        return values
    
    def search(self,event):
        def write(e,string):
            if ";" in string:
                newstring = ";".join(string.split(";")[:-1])+";"
                e.widget.insert(0,newstring)
            e.widget.select_range("end","end")
            return
        w = event.widget
        string = w.get()
        search_word = string.split(";")[-1].lower()
        iterator = zip(self.materials,self.materials_lower)
        w["values"] = [mat for mat,e in iterator if search_word in e]
        w.bind("<<ComboboxSelected>>",lambda e: write(e,string))
        w.tk.call('ttk::combobox::Post', w)
        return
    
    def shortcuts(self,event,assist=False):
        
        for iteration,row in enumerate(self.entries):
            try:
                ii = iteration
                jj = row.index(event.widget)
                break
            except: pass
        self.selection = [ii,jj]
        self.structure = self.get()
        shortcut = event.keysym
        if assist:
            if assist == "Control":
                if shortcut in ["n","N"]: self.new()
                if shortcut in ["r","R"]: self.reverse()
                if shortcut in ["d","D"]: self.duplicate()
                if shortcut in ["z","Z"]: self.undo()
                if shortcut in ["y","Y"]: self.redo()
                if shortcut == "Return": self.App.calculate()
                if shortcut == "Up": self.move("up")
                if shortcut == "Down": self.move("down")
            if assist == "Alt":
                if shortcut in ["s","S"]: self.save()
                if shortcut in ["o","O"]: self.load()
                if shortcut in ["c","C"]: self.App.config()
                if shortcut == "Return":
                    w = event.widget
                    w["values"] = self.materials
                    w.tk.call('ttk::combobox::Post', w)
            if assist == "Control-Shift":
                if shortcut in ["c","C"]: self.makeCavity()
                if shortcut in ["e","E"]: self.makePeriodic()
                if shortcut in ["i","I"]: self.makeInducedTransmission()
                if shortcut in ["q","Q"]: self.makeQuadWave()
                if shortcut in ["v","V"]: self.makeVariable()
                if shortcut == "Return": self.App.FIT()
        elif type(event.num) is not int:
            if shortcut in ["Up","Down","Left","Right",]:
                if shortcut == "Up":
                    self.selection[0] -= 1
                elif shortcut == "Down":
                    self.selection[0] += 1
                elif shortcut == "Left":
                    self.selection[1] -= 1
                elif shortcut == "Right":
                    self.selection[1] += 1
                ii,jj = self.selection
                try: selectedWidget = self.entries[ii][jj]
                except:
                    if ii>=len(self.entries): self.selection[0] = 0
                    else: self.selection[1] = 0
                    ii,jj = self.selection
                    selectedWidget = self.entries[ii][jj]
                try: selectedWidget.focus_set()
                except: print("Not available")
                selectedWidget.select_range(0,"end")
            if shortcut in ["Insert","Delete",]:
                if shortcut == "Insert": self.add()
                if shortcut == "Delete": self.delete()
            elif shortcut in ["Return"] and self.selection[1]==1:
                self.search(event)
        elif event.num==1:
            if self.selection[1]==1 and event.x>105:
                event.widget.event_generate("<Alt-Return>")
        return
        
class Selectors():

    def __init__(self,masterApp,frame,fontsize=14):

        self.App = masterApp
        self.mainframe = tk.Frame(frame,relief="solid",bd=1.5)
        self.fontsize = fontsize
        self.modenum,self.focusnum,self.polarnum = 0,0,0
        self.modevar,self.focusvar,self.polarvar = tk.IntVar(),tk.IntVar(),tk.IntVar()
        self.modes = ["RT","Phase Shift","CIE","Ellipsometry","Admittance","Efield","Absorb","Optical Constants"]
        self.focuses = ["Transmittance","Reflectance","BackReflectance",]
        self.polares = ["s-wave","mean","p-wave",]
        self.build()

        return

    def build(self):

        def Filter():
            if self.App.mode == "Efield":
                self.focuses = ["Amplitude","Intensity",]
                if self.focusnum == 0: self.polares = ["X","Y","Z","Total",]
                else: self.polares = ["s-wave","mean","p-wave",]
            elif self.App.mode in ["RT","CIE",] and self.focusnum>2: self.focusnum=0
            elif self.App.mode in ["Efield",] and self.focusnum>1: self.focusnum=0
            elif self.App.mode in ["RT","CIE","Absorb",] and self.polarnum>2: self.polarnum=0
            elif self.App.mode == "Absorb":
                self.focuses = [layer[0] for layer in self.App.structure if layer[5]]
                self.focuses.append("Total")
                self.polares = ["s-wave","mean","p-wave",]
            elif self.App.mode == "Optical Constants":
                self.focuses = [layer[0] for layer in self.App.structure]
                self.focuses.append("Total")
                self.polares = ["n","k","both",]
            elif self.App.mode == "Ellipsometry":
                self.focuses = ["Psi","Delta","Both",]
                self.polares = ["ISO","ISO+depol",]
            elif self.App.mode == "Admittance":
                self.focuses = ["iso-phase On","iso-phase Off",]
                self.polares = ["Text On","Text Off",]
            else:
                self.focuses = ["Transmittance","Reflectance","BackReflectance",]
                self.polares = ["s-wave","mean","p-wave",]

        def selectmode():
            self.modenum = self.modevar.get()
            self.focusnum = self.focusvar.get()
            self.polarnum = self.polarvar.get()
            self.App.mode = self.modes[self.modenum]
            Filter()
            try:
                focus = self.focuses[self.focusnum]
                if self.App.mode in ["Absorb","Optical Constants"]:
                    if focus != "Total": focus = self.focusnum
                self.App.focus = focus
            except: self.App.focus = self.focuses[0]
            try: self.App.polar = self.polares[self.polarnum]
            except: self.App.polar = self.polares[0]
            Norm = min([len(self.polares)-1,2])
            wt,wr,wbr = 0,0,0
            weights = [wt,wr,wbr]
            try: weights[self.focusnum] = 1
            except: pass
            wp = self.polarnum/Norm; ws = 1-wp
            self.App.weights = [*weights,ws,wp]
            self.build()
            self.App.updateGraphics(self.App.mode,)
            self.App.app_info.set("Now Ready")
            return

        def createbuttons(frame,modevar,modes,width=8,height=1,font=("",self.fontsize,""),):
            buttons = []
            for ii,mode in enumerate(modes):
                rdbtn = tk.Radiobutton(frame,width=width,height=height,text=mode,
                                        command=lambda:selectmode(),indicatoron=False,
                                        value=ii,variable=modevar,anchor="w",font=font,)
                rdbtn.pack()
                buttons.append(rdbtn)
            return buttons

        for widget in self.mainframe.pack_slaves(): widget.destroy()
        modeframe = tk.Frame(self.mainframe)
        focusframe = tk.Frame(self.mainframe)
        polarframe = tk.Frame(self.mainframe)
        title = tk.Label(self.mainframe,text="Graph Type",font=("",int(1.2*self.fontsize),"bold"),)
        title.pack()
        Filter()
        modebuttons = createbuttons(modeframe,self.modevar,self.modes,width=10,font=("",self.fontsize,"bold"),)
        focusbuttons = createbuttons(focusframe,self.focusvar,self.focuses,width=12,font=("",self.fontsize,""),)
        polarbuttons = createbuttons(polarframe,self.polarvar,self.polares,width=7,font=("",self.fontsize,""),)

        modeframe.pack(side="left")
        focusframe.pack(side="left")
        polarframe.pack(side="left")
        try: modebuttons[self.modenum].select()
        except: modebuttons[0].select()
        try: focusbuttons[self.focusnum].select()
        except: focusbuttons[0].select()
        try: polarbuttons[self.polarnum].select()
        except: polarbuttons[0].select()
        return
    
    def update(self):
        self.build()
        return
    
    def pack(self,*args,**kwrg):
        self.mainframe.pack(*args,**kwrg)
        return

class InducedTransmissionFilter():
    
    def __init__(self,App,incident,exit,font=("calibri",12,""),):
        self.App = App
        self.root = tk.Toplevel(self.App.root)
        self.font = font
        self.mainframe = tk.Frame(self.root)
        self.mainframe.pack()
        self.refdata = dict(self.App.refdata)
        self.materials = sorted(self.refdata.keys())
        self.wvl = self.App.wvl
        self.incident_index = self.refdata[incident][0](self.wvl,0) - 1j*self.refdata[incident][1](self.wvl,0)
        self.exit_index = self.refdata[exit][0](self.wvl,0) - 1j*self.refdata[exit][1](self.wvl,0)
        self.data = []
        self.createUI()
        return
    
    def createUI(self):
        for widget in self.mainframe.pack_slaves(): widget.destroy()
        
        inputframe = tk.Frame(self.mainframe)
        labelframe = tk.Frame(inputframe)
        entryframe = tk.Frame(inputframe)
        btnframe = tk.Frame(self.mainframe)

        label1 = tk.Label(labelframe,text="Absorber Material",font=self.font)
        label2 = tk.Label(labelframe,text="Absorber Thickness (nm)",font=self.font)
        label3 = tk.Label(labelframe,text="Matching Material",font=self.font)
        self.labels = [label1,label2,label3]
        for label in self.labels: label.pack()
        labelframe.pack(side="left")

        entry1 = ttk.Combobox(entryframe,width=10,height=10,
                              values=self.materials,font=self.font,)
        entry1.current(0)
        entry2 = tk.Entry(entryframe,width=12,font=self.font,)
        entry3 = ttk.Combobox(entryframe,width=10,height=10,
                              values=self.materials,font=self.font,)
        entry3.current(0)
        self.entries = [entry1,entry2,entry3]
        for entry in self.entries: entry.pack()
        entryframe.pack(side="left")

        Button_create = tk.Button(btnframe,text="Create",
                                  command=lambda: self.CLOSE(None,True))
        Button_create.pack(side="left")
        Button_cancel = tk.Button(btnframe,text="Cancel",
                                  command=lambda: self.CLOSE(None,False))
        Button_cancel.pack(side="left")

        inputframe.pack()
        btnframe.pack()
        self.root.bind("<Return>",lambda event: self.CLOSE(event,True))
        self.root.bind("<Escape>",lambda event: self.CLOSE(event,False))
        self.root.protocol("WM_DELETE_WINDOW",lambda:self.CLOSE())
        entry1.focus_set()
        self.mainframe.mainloop()
        return
    
    def update(self):
        wvl0 = self.App.wvl
        refindex = []; materials = []
        for ii,entry in enumerate(self.entries):
            if ii in [0,2]:
                material = entry.get()
                materials.append(material)
                refdata = self.refdata[material]
                refindex.append([refdata[0](self.wvl,0),refdata[1](self.wvl,0),])
            elif ii in [1]:
                Thickness = float(entry.get())
        N_film,N_match = refindex

        n,k = N_film
        y = n-1j*k
        delta = 2*np.pi*y*Thickness/wvl0
        sinD,cosD = np.sin(delta),np.cos(delta)

        alpha,beta = delta.real,-delta.imag
        q = abs(sinD)**2
        r = np.sin(alpha)*np.cos(alpha)
        s = np.sinh(beta)*np.cosh(beta)
        rr,ss = n*s-k*r,n*s+k*r
        Z = n*k*q/rr
        X = np.sqrt(abs((ss/rr)*(n**2+k**2)-Z**2))

        Yo = self.incident_index
        Ysub = self.exit_index
        Yexit = X+1j*Z
        Yin = (1j*y*sinD+Yexit*cosD)/(cosD+1j*Yexit*(sinD/y))
        print(f"Yo: {Yo}\n,Ysub: {Ysub}\n,Yexit: {Yexit}\n,Yin: {Yin}\n")

        n,k = N_match
        y = n-1j*k
        try: opticalD_match_up = np.arctan((Yexit-Ysub)/(1j*(y-Yexit*Ysub/y)))
        except: opticalD_match_up = np.pi/2
        try: opticalD_match_down = np.arctan((Yo-Yin)/(1j*(y-Yo*Yin/y)))
        except: opticalD_match_down = np.pi/2
        match_up = ((wvl0/(2*np.pi*n))*opticalD_match_up)
        match_down = ((wvl0/(2*np.pi*n))*opticalD_match_down)
        if match_up<0: match_up += wvl0/(4*n)
        if match_down<0: match_down += wvl0/(4*n)
        material_film,material_match = materials
        self.data = [[material_match,match_up.real[0],[0,None],0,[],1],
                     [material_film,Thickness,[0,None],0,[],1],
                     [material_match,match_down.real[0],[0,None],0,[],1],]
        return
    
    def focus_set(self,widget):
        widget.focus_set()
        try: widget.select_range(0,"end")
        except: pass
        return
    
    def get(self):
        return self.data
    
    def CLOSE(self,event=None,isupdate="ask"):
        if isupdate=="ask":
            title = "Create Structure"
            question = "Do you want to create the induced transmission filter?"
            isupdate = tk.messagebox.askokcancel(title,question)
        if isupdate:
            self.update()
        else:
            self.data = []
        self.mainframe.quit()
        return
    
    def terminate(self):
        self.root.destroy()
        return

class PeriodicSettings():

    def __init__(self,master,font=("calibri",12,""),cur_layer=["TiO2",0,[0,None],0,[],1]):
        self.master = master
        self.root = tk.Toplevel(self.master.root)
        self.font = font
        self.mainframe = tk.Frame(self.root)
        self.mainframe.pack()
        self.data = [cur_layer]
        self.createUI()
        return
    
    def createUI(self):
        for widget in self.mainframe.pack_slaves(): widget.destroy()
        topframe = tk.Frame(self.mainframe)
        topframe.pack()
        tk.Label(topframe,text="Number of Periods:",font=self.font).pack(side="left")
        self.number = tk.Entry(topframe,font=self.font)
        self.number.insert(0,"1")
        self.number.select_range(0,"end")
        self.number.pack(side="left")
        self.table = StructureTable(self.master,self.mainframe,structure=self.data,
                                    columns=["Material","Thickness","FitRange","DepoAngle","FitRange","Coherence",],
                                    font=self.font,refdata=self.master.refdata,showmediums=False,)
        self.table.pack()
        self.root.bind("<Home>",lambda event: self.focus_set(self.number))
        self.root.bind("<End>",lambda event: self.focus_set(self.table))
        self.root.bind("<Control-Return>",lambda event: self.CLOSE(event,True))
        self.root.bind("<Escape>",lambda event: self.CLOSE(event,False))
        self.root.protocol("WM_DELETE_WINDOW",lambda:self.CLOSE())
        self.number.focus_set()
        self.root.mainloop()
        return
    
    def focus_set(self,widget):
        widget.focus_set()
        try: widget.select_range(0,"end")
        except: pass
        return
    
    def update(self):
        newdata = []
        for ii in range(int(self.number.get())):
            newdata.extend(self.table.get())
        self.data = newdata
        return
    
    def get(self):
        return self.data
    
    def CLOSE(self,event=None,isupdate="ask"):
        if isupdate=="ask":
            title = "Create Structure"
            question = "Do you want to create the periodic structure?"
            isupdate = tk.messagebox.askokcancel(title,question)
        if isupdate:
            self.update()
        else:
            self.data = []
        self.root.quit()
        return
    
    def terminate(self):
        self.root.destroy()
        return


class Info():
    
    def __init__(self,master,title="  ",message="  ",width=30,):
        self.root = tk.Toplevel(master)
        self.title = tk.Label(self.root,text=title,width=width,
                             font=("Helvetica",12,"bold"),anchor="center",)
        self.message = tk.Label(self.root,text=message,width=width,
                             font=("Helvetica",10,""),anchor="w",justify="left")
        self.extras = []
        self.data   = []
        self.createUI()
        return
    
    def createUI(self):
        tk.Label(self.root,text="    ").pack()
        self.title.pack()
        self.message.pack()
        for e in self.extras: e.pack()
        tk.Label(self.root,text="    ").pack()
        self.PAUSE()
        return
    
    def updatetitle(self,title):
        self.title.config(text=title)
        self.PAUSE()
        return
    
    def update(self,message):
        self.message.config(text=message)
        self.PAUSE()
        return
    
    def _add_text(self,text=" ",width=30,font=("Helvetica",12,"bold"),anchor="center",*args,**kwrg):
        e = tk.Message(self.root,text=text,width=width,font=font,anchor=anchor,*args,**kwrg)
        self.extras.append(e)
        return
    
    def _add_yes_no(self,width=10,font=("Helvetica",12,"bold"),
                    fx_y=None,fx_y_arg=[None],fx_n=None,fx_n_arg=[None],
                    *args,**kwrg):
        default_fx = lambda: self.terminate()
        if fx_y is None: yes = default_fx
        else: yes = lambda: self.terminate_after(fx_y,fx_y_arg)
        if fx_n is None: no = default_fx
        else: no = lambda: self.terminate_after(fx_n,fx_n_arg)
        frame = tk.Frame(self.root)
        y = tk.Button(frame,text="yes",width=width,font=font,anchor="center",
                      command=yes,*args,**kwrg)
        n = tk.Button(frame,text="no",width=width,font=font,anchor="center",
                      command=no,*args,**kwrg)
        y.pack(side="left")
        n.pack(side="right")
        self.extras.append(frame)
        return
    
    def _add_entry(self,text=" ",width=30,font=("Helvetica",12,"bold"),anchor="center",*args,**kwrg):
        e = tk.Entry(self.root,width=width,font=font,*args,**kwrg)
        e.insert(0,text)
        self.data.append(e.get)
        self.extras.append(e)
        return
    
    def terminate_after(self,func,args):
        with open("email_for_bug_report.dat","w") as f:
            for method in self.data: f.write(method())
        func(*args)
        self.terminate()
        return
    
    def terminate(self):
        self.root.withdraw()
        self.PAUSE()
        return
    
    def PAUSE(self,waittime=1):
        var = tk.IntVar()
        self.root.after(waittime,var.set,1)
        self.root.wait_variable(var)
        return

class Configuration():
    
    def __init__(self,masterApp,font,filepath=".\\",debug_flg=False):
        
        self.App = masterApp
        self.master = masterApp.root
        self.filepath = filepath
        
        font,size,style = font
        self.font = (font,size,style)
        self.boldfont = (font,size+2,"bold")
        
        self.fittypes = []
        self.fitangles = []
        self.isupdated = False
        self.ask = False
        configs = self.get()
        
        init_angle = configs["angleconfig"]
        init_wvlth = configs["wvlthconfig"]
        init_color = [configs["XYZrate"]]
        init_angle.insert(0,"Angle Settings")
        init_wvlth.insert(0,"Wavelength Settings")
        init_color.insert(0,"Fit Settings")

        if self.App.targetspectra_raw: self.fitmode = "Spectra"
        else: self.fitmode = "Color"
        
        self.developer_mode = configs["developer_mode"]
        self.initialvalues = init_angle,init_wvlth,init_color
        self.wvlth_interp_rng = configs["wvlth_interp_rng"]
        self.focusColor = configs["targetcolor"]
        self.XYZrate = configs["XYZrate"]
        self.fitangleEntries = []
        self.root = tk.Toplevel(self.master)
        self.mainframe = tk.Frame(self.root)

        self.createUI(debug_flg)
        
        return
        
    def createUI(self,debug_flg=False):
        
        def create_config(frame,mode=""):
            mainframe = tk.Frame(frame)
            anglelabels = ["Angle Settings","minimum","maximum","interval",]
            wvlthlabels = ["Wavelength Settings","minimum","maximum","interval",]
            fitterlabels = ["Fit Settings","XYZ Ratio"]
            angleinitial = self.initialvalues[0]
            wvlthinitial = self.initialvalues[1]
            fitterinitial = self.initialvalues[2]
            if mode=="angle": labels = anglelabels; initials = angleinitial; width=8
            elif mode=="wvlth": labels = wvlthlabels; initials = wvlthinitial; width=8
            elif mode=="fitter": labels = fitterlabels; initials = fitterinitial; width=16
            for ii,(label,initial) in enumerate(zip(labels,initials)):
                row = tk.Frame(mainframe)
                row.pack()
                if ii:
                    tk.Label(row,text=label,font=self.font,width=width).pack(side="left")
                    e = tk.Entry(row,font=self.font,width=8)
                    e.insert(0,initial)
                    e.pack(side="left")
                    if mode=="angle": self.angleEntries.append(e)
                    elif mode=="wvlth": self.wvlthEntries.append(e)
                    elif mode=="fitter": self.fitterEntries = [e]
                else:
                    tk.Label(row,text=label,font=self.boldfont).pack(side="left")
            return mainframe
        
        def createSelectors(frame,):
            
            focuses = {"BLUE":[0,0,1],"GREEN":[0,1,0],"RED":[1,0,0],"WHITE":[1,1,1],"AR":"AR",
                       "YELLOW":[1,1,0],"MAGENTA":[1,0,1],"CYAN":[0,1,1],"BLACK":[0,0,0],"HR":"HR",}
            focuskeys = list(focuses.keys())
        
            def selectmode():
                self.focusColor = focuskeys[focusvar.get()]
                return
            mainframe = tk.Frame(frame)
            focusframe = tk.Frame(mainframe)
            focusvar = tk.IntVar()
        
            focusbuttons = []
            for ii,focus in enumerate(focuskeys):
                if ii in [0,5]: col = tk.Frame(focusframe); col.pack(side="left")
                rdbtn = tk.Radiobutton(col,width=20,text=focus,command=selectmode,indicatoron=False,
                                    value=ii,variable=focusvar,anchor="w",font=self.font,)
                rdbtn.pack()
                focusbuttons.append(rdbtn)
            focusbuttons[focuskeys.index(self.focusColor)].select()
            focusframe.pack()
            return mainframe
        
        def createFitConfig(frame):
            
            def selectmode():
                for widget in tabframe.pack_slaves(): widget.destroy()
                self.fitmode = fitmodes[tabvar.get()]
                build_tab()
                return
            
            def build_tab():
                for widget in tabframe.pack_slaves(): widget.destroy()
                fitterconfig = create_config(tabframe,"fitter",)
                if self.fitmode == "Color":
                    fitterconfig.pack()
                    fitterconfig = createSelectors(tabframe)
                    fitterconfig.pack()
                elif self.fitmode == "Spectra":
                    fitdata = self.App.targetspectra_raw
                    fittype = ["Ts","Tp","Tm","Rs","Rp","Rm","bRs","bRp","bRm","dep","iso"]
                    self.fittypes = []
                    self.fitangles = []
                    for ii,data in enumerate(fitdata):
                        row = tk.Frame(tabframe,)
                        tk.Label(row,text=data[0]).pack(side="left")
                        CBx = ttk.Combobox(row,width=3,values=fittype)
                        CBx.pack(side="right")
                        try: index = fittype.index(data[2])
                        except: index = 0
                        CBx.current(index)
                        self.fittypes.append(CBx)
                        Ent = tk.Entry(row,width=3,)
                        try: Ent.insert(0,data[3])
                        except: Ent.insert(0,"0")
                        Ent.pack(side="right")
                        self.fitangles.append(Ent)
                        row.pack(fill="x")
                    row = tk.Frame(tabframe,)
                    tk.Button(row,text="import",
                              command=self.importdata,).pack(side="left")
                    tk.Button(row,text="drop all",
                              command=self.dropdata,).pack(side="left")
                    row.pack()
                return

            mainframe = tk.Frame(frame,relief="sunken",bd=1)
            selectorframe = tk.Frame(mainframe)
            tabframe = tk.Frame(mainframe)

            tabvar = tk.IntVar()
            fitmodes = ["Color","Spectra"]
            index = fitmodes.index(self.fitmode)
            for ii,tab in enumerate(fitmodes):
                rdbtn = tk.Radiobutton(selectorframe,width=20,text=tab,command=selectmode,indicatoron=False,
                                    value=ii,variable=tabvar,anchor="w",font=self.font,)
                rdbtn.pack(side="left")
                if ii == index: rdbtn.select()

            build_tab()
            selectorframe.pack(fill="x")
            tabframe.pack(fill="x")

            return mainframe,build_tab

        for widget in self.mainframe.pack_slaves(): widget.destroy()

        self.angleEntries = []
        self.wvlthEntries = []

        upperframe = tk.Frame(self.mainframe)
        lowerframe = tk.Frame(self.mainframe)

        angleconfig = create_config(upperframe,"angle",)
        angleconfig.pack(side="left")
        wvlthconfig = create_config(upperframe,"wvlth",)
        wvlthconfig.pack(side="left")
        fitconfig,self.update_fitconfig = createFitConfig(lowerframe)
        fitconfig.pack()
        
        self.mainframe.pack()
        upperframe.pack()
        lowerframe.pack()
        
        self.angleEntries[0].focus_set()
        self.angleEntries[0].select_range(0,"end")
        self.root.bind("<Return>",lambda event:self.CLOSE(event,True))
        self.root.bind("<Escape>",lambda event:self.CLOSE(event,False))
        self.root.protocol("WM_DELETE_WINDOW",lambda:self.CLOSE())
        if not debug_flg: self.root.mainloop()
        return
    
    def update(self,ask=False):
        "ask is forced update"
        angle = []
        wvlth = []
        fitter = []
        for e in self.angleEntries:
            try: angle.append(float(e.get()))
            except: angle.append("default")
        for e in self.wvlthEntries:
            try: wvlth.append(float(e.get()))
            except: wvlth.append("default")
        for e in self.fitterEntries:
            value = e.get()
            try: fitter.append(float(value))
            except:
                if value in ["lab","Lab","LAB","CIELAB",]: fitter.append("CIELAB")
                elif value.lower() in ["ok","oklab",]: fitter.append("OKLAB")
                else: fitter.append("default")
        
        new_min,new_max = wvlth[:-1]
        if ask:
            self.wvlth_interp_rng = [new_min,new_max]
            interp_token = True
        else:
            old_min,old_max = self.wvlth_interp_rng
            self.wvlth_interp_rng = [min(new_min,old_min),max(new_max,old_max)]
            interp_token = new_min<old_min or new_max>old_max

        with open(self.filepath+"Configuration.dat","w") as f:
            f.write(str(self.developer_mode)+"\n")
            start,end,interval = angle
            if start=="default": start = 0
            if end=="default": end = 0
            if interval=="default": interval = 10
            f.write(str(start)+"\n"+str(end)+"\n"+str(interval)+"\n")
            start,end,interval = wvlth
            if start=="default": start = 350
            if end=="default": end = 750
            if interval=="default": interval = 1
            f.write(str(start)+"\n"+str(end)+"\n"+str(interval)+"\n")
            start,end = self.wvlth_interp_rng
            if start=="default": start = 350
            if end=="default": end = 750
            f.write(str(start)+"\n"+str(end)+"\n")
            XYZrate = fitter[0]
            if XYZrate=="default": XYZrate = self.App.XYZrate
            f.write(str(XYZrate)+"\n")
            f.write(str(self.focusColor)+"\n")
        return interp_token

    def get(self):
        try:
            with open(self.filepath+"Configuration.dat") as f:
                read = f.read().splitlines()
                developer_mode = int(read.pop(0))
                angleconfig = read[0:3]
                wvlthconfig,wvlth_interp_rng = read[3:6],read[6:8]
                XYZrate,targetcolor = read[8],read[9]
        except ValueError:
            developer_mode = 0
            angleconfig = [0,0,10]
            wvlthconfig,wvlth_interp_rng = [350,750,1],[350,750]
            XYZrate,targetcolor = "CIELAB","BLUE"
        
        angle = [float(e) for e in angleconfig]
        wvlth = [float(e) for e in wvlthconfig]
        wvlth_interp_rng = [float(e) for e in wvlth_interp_rng]

        fittype = [e.get() for e in self.fittypes]
        angles = [e.get() for e in self.fitangles]
        results = {
            "developer_mode": developer_mode,
            "angleconfig": angle,
            "wvlthconfig": wvlth,
            "wvlth_interp_rng": wvlth_interp_rng,
            "XYZrate": XYZrate,
            "targetcolor": targetcolor,
            "updated_token": self.isupdated,
            "ask_token": self.ask,
            "fittype": fittype,
            "fitangles": angles,
        }
        return results

    def importdata(self):
        # txt: integration sphere
        # dat: ellipsometer
        # csv: self made (normal incidence)
        # ang: self made (oblique incidence)
        filename = tk.filedialog.askopenfilename(initialdir=self.App.datapathC,title="select a file",
                                            filetypes =(("csv files","*.csv"),("dat files","*.dat"),("angular data files","*.ang"),("all files","*.*")))
        if filename:
            data = []
            filetype = filename.split(".")[-1]
            if filetype in ["csv","txt"]:
                data_row = [filename.split(".")[0]]
                with open(filename,'r') as f:
                    if filetype=="csv":
                        reader = csv.reader(f)
                        normalizer = 1
                    elif filetype=="txt":
                        reader = f.read().splitlines()
                        normalizer = 100
                    xx,yy = [],[]
                    for ii in reader:
                        try:
                            if len(ii) != 2: ii = ii.split("\t")
                            if any(ii)=="": raise Exception
                            xx.append(float(ii[0]))
                            yy.append(float(ii[1])/normalizer)
                        except: pass
                try: fq = interp1d(xx,yy,kind="quadratic",fill_value="extrapolate")
                except: print(xx,yy)
                data_row.append(np.array([self.App.wvlths,fq(self.App.wvlths)],dtype=np.float).T)
                Sweeper = ["Ts","Tp","Tm","Rs","Rp","Rm","bRs","bRp","bRm","dep","iso"]
                for fittype in reversed(Sweeper):
                    if fittype in data_row[0]: data_row.extend([fittype,0,])
                data.append(data_row)
            elif filetype == "dat":
                dataname = filename.split(".")[0]
                with open(filename,'r',encoding='utf-8') as f:
                    read = f.read().splitlines()
                    data_measure_mthd = None
                    for ii,row in enumerate(read):
                        if "VASEmethod" in row:
                            fittype = "PsiDelta"
                            if "Isotropic+Depolarization" in row:
                                data_measure_mthd = "dep"
                            else:
                                data_measure_mthd = "iso"
                        elif "RTmethod" in row: fittype = "RT"
                        elif "nm" in row:
                            threshold = ii+1
                            convertx = lambda x: x
                            break
                        elif "eV" in row:
                            threshold = ii+1
                            convertx = lambda x: 1.2398e3/x
                            break
                    datatypes,datalists = [],[]
                    if fittype == "PsiDelta":
                        if data_measure_mthd == "dep":
                            for ii,row in enumerate(reversed(read[threshold:])):
                                if row.split("\t")[0]!="dpolE": end = ii; break
                            if end!=0: read = read[:-end]
                        for row in read[threshold:]:
                            cols = row.split("\t")
                            wvl = convertx(float(cols[0]))
                            psi,dta = np.deg2rad(float(cols[2])),np.deg2rad(float(cols[3]))
                            rho = np.array(np.tan(psi)*np.exp(1j*dta))
                            if [data_measure_mthd,cols[1]] in datatypes:
                                datalist[0].append(wvl)
                                datalist[1].append(rho)
                            else:
                                try: datalists.append(datalist)
                                except NameError: pass
                                datatypes.append([data_measure_mthd,cols[1]])
                                datalist = [[wvl],[rho],]
                        datalists.append(datalist)
                    elif fittype == "RT":
                        for row in read[threshold:]:
                            cols = row.split("\t")
                            if [cols[0][::-1],cols[2],] in datatypes:
                                datalist[0].append(float(cols[1]))
                                datalist[1].append(float(cols[3]))
                            else:
                                try: datalists.append(datalist)
                                except NameError: pass
                                datatypes.append([cols[0][::-1],cols[2],])
                                datalist = [[float(cols[1])],[float(cols[3])],]
                        datalists.append(datalist)
                    for datatype,datum in zip(datatypes,datalists):
                        if datatype[0] in ["dep","iso"]: dtype = np.complex
                        else: dtype = np.float
                        fq = interp1d(*datum,kind="quadratic",fill_value="extrapolate")
                        data_row = [dataname]
                        data_row.append(np.array([self.App.wvlths,fq(self.App.wvlths)],dtype=dtype).T)
                        data_row.extend(datatype)
                        data.append(data_row)
                # with open(filename,'r',encoding='utf-8') as f:
                #     read = f.read().splitlines()
                #     for ii,row in enumerate(read):
                #         if "VASEmethod" in row:
                #             data["type"] = "PsiDelta"
                #             if "Isotropic+Depolarization" in row: data["measure_method"] = "dep"
                #             else: data["measure_method"] = "iso"
                #         elif "RTmethod" in row: data["type"] = "RT"
                #         elif "nm" in row:
                #             threshold = ii+1
                #             convertx = lambda x: x
                #             break
                #         elif "eV" in row:
                #             threshold = ii+1
                #             convertx = lambda x: 1.2398e3/x
                #             break
                #     if data["type"] == "PsiDelta":
                #         data["data"] = {"Psi":[],"Delta":[],"Rho":[],}
                #         for row in read[threshold:]:
                #             cols = row.split("\t")
                #             if len(cols) == 6:
                #                 wvlth,angle,psi,dta,err1,err2 = cols
                #                 data["wavelength"].append(convertx(float(wvlth)))
                #                 data["angle"].append(float(angle))
                #                 data["data"]["Psi"].append(np.deg2rad(float(psi)))
                #                 data["data"]["Delta"].append(np.deg2rad(float(dta)))
                #     elif data["type"] == "RT":
                #         data["data"] = {}
                #         for row in read[threshold:]:
                #             cols = row.split("\t")
                #             if len(cols) == 5:
                #                 label,wvlth,angle,val,err = cols
                #                 data["wavelength"].append(convertx(float(wvlth)))
                #                 data["angle"].append(float(angle))
                #                 data["data"][f"{label[::-1]}"].append(val)
            elif filetype == "ang":
                dataname = filename.split(".")[0]
                dataname_split = dataname.split("_")
                Sweeper = ["Ts","Tp","Tm","Rs","Rp","Rm","bRs","bRp","bRm","dep","iso"]
                for fittype_try in reversed(Sweeper):
                    if fittype_try in dataname_split:
                        fittype = fittype_try
                        break
                    else: fittype = "Ts"
                with open(filename,'r',encoding='utf-8') as f:
                    read = csv.reader(f)
                    wvl,datalists = [],[]
                    for ii,row in enumerate(read):
                        if ii:
                            wvl.append(float(row[0]))
                            datalists.append(row[1:])
                        else:
                            angles = row[1:]
                    datalists = np.array(datalists,dtype=float).T
                    for ang,datum in zip(angles,datalists):
                        fq = interp1d(wvl,datum,kind="quadratic",fill_value="extrapolate")
                        data_row = [dataname]
                        data_row.append(np.array([self.App.wvlths,fq(self.App.wvlths)],dtype=np.float).T)
                        data_row.extend([fittype,ang])
                        data.append(data_row)
            self.App.targetspectra_raw.extend(data)
            self.update_fitconfig()
        self.root.focus_set()

        return
    
    def dropdata(self,Btn=None):
        self.App.targetspectra_raw = []
        self.update_fitconfig()
        return
    
    def CLOSE(self,event=None,isupdate="ask"):
        ask = isupdate=="ask"
        if ask:
            title = "Forced Interpolation"
            question = "Do you want to force interpolation anyway?"\
                       "\n(This is necessary when new material is added to Data folder.)"
            ask = tk.messagebox.askokcancel(title,question)
        if isupdate: ask = self.update(ask=ask)
        self.ask = ask
        self.isupdated = isupdate or ask
        self.root.quit()
        return
    
    def terminate(self,):
        self.root.withdraw()
        return



# def TMM_sub(structure,refdata,angle0=0,):
    
#     def getCoefs(N=-1,):
#         Ea_s,Ea_p,Eb_s,Eb_p,Ec_s,Ec_p = getElements(N)
#         try: ts = 1/Ea_s
#         except: ts = 0; print("OVERFLOW!")
#         try: tp = 1/Ea_p
#         except: tp = 0; print("OVERFLOW!")
#         try: rs = Ec_s/Ea_s
#         except: rs = 0; print("OVERFLOW!")
#         try: rp = Ec_p/Ea_p
#         except: rp = 0; print("OVERFLOW!")
#         try: brs = -Eb_s/Ea_s
#         except: brs = 0; print("OVERFLOW!")
#         try: brp = -Eb_p/Ea_p
#         except: brp = 0; print("OVERFLOW!")
#         return ts,tp,rs,rp,brs,brp
    
#     def getElements(N,):
#         Ea_s,Ea_p = [],[]
#         Eb_s,Eb_p = [],[]
#         Ec_s,Ec_p = [],[]
#         for e in TransferMat_s:
#             a = e[N][0,0]
#             b = e[N][0,1]
#             c = e[N][1,0]
#             Ea_s.append(a)
#             Eb_s.append(b)
#             Ec_s.append(c)
#         for e in TransferMat_p:
#             a = e[N][0,0]
#             b = e[N][0,1]
#             c = e[N][1,0]
#             Ea_p.append(a)
#             Eb_p.append(b)
#             Ec_p.append(c)
#         Ea_s = np.array(Ea_s)
#         Ea_p = np.array(Ea_p)
#         Eb_s = np.array(Eb_s)
#         Eb_p = np.array(Eb_p)
#         Ec_s = np.array(Ec_s)
#         Ec_p = np.array(Ec_p)
#         return Ea_s,Ea_p,Eb_s,Eb_p,Ec_s,Ec_p

#     structure_temp = []
#     for layer in structure:
#         layer[1] = 0
#     matrices_s,matrices_p = [],[]
#     wavelength = refdata[0][1][:,0]
#     for ii,layer in enumerate(structure):
#         if ii:
#             structure_temp.append(layer)
#             print(structure_temp)
#             res = TransferMatrixMethod(structure_temp,refdata,angle0,)
#             [ts,tp,rs,rp,brs,brp],[ratio_s,ratio_p] = res
#             Ts,Rs = abs(ts**2)*ratio_s,abs(rs**2)
#             Tp,Rp = abs(tp**2)*ratio_p,abs(rp**2)
#             matrix_s = [np.array([[1,R],[R,1]])/T for T,R in zip(Ts,Rs)]
#             matrix_p = [np.array([[1,R],[R,1]])/T for T,R in zip(Tp,Rp)]
#             matrices_s.append(matrix_s)
#             matrices_p.append(matrix_p)
#             structure_temp.pop(0)
#         else: structure_temp.append(layer)
#     TransferMat_s = []; TransferMat_p = []
#     for ii in range(len(wavelength)):
#         sM = np.eye(2); pM = np.eye(2)
#         row_sM = [sM]; row_pM = [pM]
#         for M_s,M_p in zip(matrices_s,matrices_p):
#             try:
#                 sM = sM@M_s[ii]
#                 pM = pM@M_p[ii]
#             except: print("OVERFLOW!")
#             row_sM.append(sM)
#             row_pM.append(pM)
#         TransferMat_s.append(row_sM)
#         TransferMat_p.append(row_pM)
    
#     final_results = getCoefs()
#     return final_results