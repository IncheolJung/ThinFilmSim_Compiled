from numpy import array,ones_like,zeros_like,sum,power,conj,sqrt,expand_dims,str_
from copy import deepcopy



def EMA(layer):
    try: layer[0] = eval(layer[0])
    except (NameError,TypeError):
        try: layer[0] = dict(layer[0])
        except ValueError: pass
    except SyntaxError: pass
    if type(layer[0]) is str:
        reader = layer[0]
        if reader[0] == "[":  reader = reader[1:]
        if reader[-1] == "]": reader = reader[:-1]
        material = [e.lstrip().rstrip() for e in reader.split(";")]
        l_kw,b_kw,g_kw = ["l","linear",],["b","bragg-pipard",],["g","glad",]
        material_0 = material[0]
        if "%" in material_0:
            kwd,Q = material_0.split("%")
            material_0 = kwd
        if material_0.lower() in l_kw:   EMA_mthd = "Linear"
        elif material_0.lower() in b_kw: EMA_mthd = "Bragg-Pippard"
        elif material_0.lower() in g_kw: EMA_mthd = "GLAD"
        else:
            print("EMA method not specified. falling back to Bragg-Pippard")
            EMA_mthd = "Bragg-Pippard"
            material.insert(0,EMA_mthd)
        
        fillfactor,depolfactor,concentration = None,None,[]
        c_bnd,f_bnd,Q_bnd = [],None,None
        if len(material)<2: return
        elif "%" in material[1]:
            _ = array([mc.split("%") for mc in material[1:]])
            material,concentration = _.T
            if EMA_mthd=="Bragg-Pippard": params = [Q,*concentration]
            elif EMA_mthd=="Linear":  params = [*concentration]
            for ii,e in enumerate(params):
                if "f" in e.lower():
                    e,bnd = e.lower().split("f")
                    try: bnd = list(eval(bnd))
                    except SyntaxError:
                        bnd = bnd+"]"
                        bnd = list(eval(bnd))
                    except (TypeError,NameError): bnd = [0.,1.]
                else: bnd = []
                params[ii] = [e,bnd]
            if EMA_mthd=="Bragg-Pippard":
                Q,bnd = params.pop(0)
                depolfactor = float(Q)
                Q_bnd = bnd
            for ii,(e,bnd) in enumerate(params):
                concentration[ii] = e
                c_bnd.append(bnd)
            concentration = concentration.astype(float)
            fillfactor = concentration[0]/sum(concentration)
            if EMA_mthd=="Bragg-Pippard":
                for bnd in c_bnd:
                    if bnd: f_bnd = bnd
                c_bnd = None
        else:
            del material[0]
            try:
                try:
                    ff,fb,qq,qb = layer[3:]
                except ValueError:
                    (ff,fb,qq),qb = layer[3:],[]
                try:
                    fillfactor,depolfactor = float(ff),float(qq)
                    f_bnd,Q_bnd = list(fb),list(qb)
                except ValueError:
                    fillfactor,depolfactor = ff,qq
                    f_bnd,Q_bnd = fb,qb
            except TypeError: pass
            try:
                try:
                    concentration = array(layer[3],dtype=float)
                    if len(concentration)==len(material)-2:
                        c = (1-array(concentration))
                        if c>0: concentration.append(c)
                        else: concentration.append(0.)
                except ValueError:
                    concentration = str(layer[3])
                    c_bnd = layer[4]
                if layer[4]:
                    if array(layer[4]).ndim==1:
                        if len(concentration)>1:
                            layer[4] = deepcopy([layer[4]]*len(material))
                    elif array(layer[4]).ndim==2:
                        if len(layer[4])!=len(material):
                            print("INVALID BOUNDARY")
                    c_bnd = layer[4]
            except TypeError:
                if fillfactor is not None:
                    concentration = [fillfactor,1.-fillfactor]
                    c_bnd = [[0.,1.,],[0.,1.,]]
        bnds = {
                "concentration": c_bnd,
                "filling_factor": f_bnd,
                "depolarization_factor": Q_bnd,
                "deposition_angle": f_bnd,
                }
        EMA_model = {
            "EMAmethod": EMA_mthd,
            "material": material,
            "concentration": concentration,
            "filling_factor": fillfactor,
            "depolarization_factor": depolfactor,
            "deposition_angle": fillfactor,
            "bnds": bnds,
            "repr": "EMA",
            "refindex": "init",
            }
        layer[0],layer[3:5] = EMA_model,(0,[])
        if type(layer[1]) is not str: layer[5] = 1
    elif type(layer[0]) is tuple:
        layer[0] = list(layer[0])
        EMA(layer)
    elif type(layer[0]) is list:
        for e in layer[0]: EMA(e)
        try:
            concentration = list(layer[3])
            if len(concentration)==len(material)-2:
                c = (1-array(concentration).sum())
                concentration.append(max(c,0))
            if layer[4]:
                if array(layer[4]).ndim==1:
                    if len(concentration)>1:
                        layer[4] = deepcopy(layer[4]*len(concentration))
                elif array(layer[4]).ndim==2:
                    if len(layer[4])!=len(concentration):
                        print("INVALID BOUNDARY")
                c_bnd = layer[4]
        except TypeError:
            concentration = ones_like(layer[0],dtype=float)
            concentration /= float(len(concentration))
            c_bnd = [[0.,1.] for e in concentration]
        EMA_model = {
            "EMAmethod": "Linear",
            "material": layer[0],
            "refindex": "init",
            "concentration": concentration,
            "bnds": {"concentration": c_bnd},
            "repr": "EMA"
            }
        EMA_model = update_repr(EMA_model)
        layer[0] = EMA_model
    return



def update_repr(material):
    for m in material["material"]:
        if type(m) is dict: update_repr(m)
    mats = []
    for m in material["material"]:
        if   type(m) is dict: newmat = m["repr"]
        elif type(m) in [str,str_]:  newmat = m
        else:
            newmat = m
            print(f"INVALID MATERIAL: {m}")
        mats.append(newmat)
    mthd = material["EMAmethod"][0]
    if material["EMAmethod"]=="Bragg-Pippard":
        if material["bnds"]["depolarization_factor"]:
            tail = "f"+str(material["bnds"]["depolarization_factor"]).replace(" ","")
        else: tail = ""
        mthd = mthd+"%"+f'{material["depolarization_factor"]:.2f}'+tail
        f = material["filling_factor"]
        cons = [f,1-f]
        bnds = [False,material["bnds"]["filling_factor"]]
    else:
        cons = material["concentration"]
        bnds = material["bnds"]["concentration"]
    mat_ratio = []
    for m,c,b in zip(mats,cons,bnds):
        if b: tail = "f"+str(b).replace(" ","")
        else: tail = ""
        mat_ratio.append(m+"%"+f"{c:.2f}"+tail)
    reprs = [mthd,*mat_ratio]
    material["repr"] = "["+";".join(reprs)+"]"
    return



def material_update_for_EMA(structure):
    structure = deepcopy(structure)
    for layer in structure:
        if type(layer[0]) is dict:
            update_repr(layer[0])
            layer[0] = layer[0]["repr"]
    return structure



def extract_refindex(material,refdata,wavelength,ang=0):

    def get_refindex(mat,ang,pol="p"):
        if pol == "p":   idx_mod = 0
        elif pol == "s": idx_mod = 2
        refindex  = refdata[mat][idx_mod+0](wavelength,ang).astype(complex)
        refindex -= 1j*refdata[mat][idx_mod+1](wavelength,ang)
        return refindex
    
    if type(material) is dict:
        if material["refindex"] == "init":
            material["refindex"] = []
            for ii,e in enumerate(material["material"]):
                material["refindex"].append(extract_refindex(e,refdata,wavelength,ang))
            EMAmethod = material["EMAmethod"]
            if EMAmethod=="Bragg-Pippard": EMA_fx = BraggPippard
            elif EMAmethod=="Linear": EMA_fx = Linear
            elif EMAmethod=="GLAD": EMA_fx = GLAD
            refindex_s,refindex_p = EMA_fx(material)
        elif array(material["refindex"]).shape == (2,len(wavelength)):
            refindex_s,refindex_p = material["refindex"]
        else:
            print(F"""
            INVALID refindex IN {material['repr']}. REINITIALIZING refindex.
            """)
            material["refindex"] = "init"
            refindex_p,refindex_s = extract_refindex(material,refdata,wavelength,ang)
        material["refindex"] = (refindex_p,refindex_s)
    elif type(material) in [str,str_]:
        try:
            refindex_s = get_refindex(material,ang,pol="s")
            refindex_p = get_refindex(material,ang,pol="p")
        except KeyError:
            try:
                material = eval(material)
                refindex_p,refindex_s = extract_refindex(material,refdata,wavelength,ang)
            except: raise Exception(f"INVALID MATERIAL: {material}")
    else:
        refindex_p,refindex_s = material
        print(type(material))
    return refindex_p,refindex_s

def BraggPippard(material):
    try:
        _ = material["material"][2]
        length = len(material["material"])
        print(f"Bragg-Pippard model error: material number is larger than 2")
        print(f"current length: {length}")
    except IndexError: pass
    n1,n2 = material["refindex"][0],material["refindex"][1]
    f,Q   = material["filling_factor"],material["depolarization_factor"]
    ep_1,ep_2 = power(conj(n1),2),power(conj(n2),2)
    ep_diff   = ep_1-ep_2
    ep_eff    = ep_2+(f*ep_diff)/(1+Q*(1-f)*(ep_diff/ep_2))
    n_eff     = conj(sqrt(ep_eff))
    return n_eff

def Linear(material):
    ref_array = array(material["refindex"],dtype=complex)
    ep_array  = ref_array**2
    c_array   = array(material["concentration"],dtype=complex)
    try:
        c_array = expand_dims(c_array/sum(c_array),axis=(1,2))
    except FloatingPointError:
        c_array = zeros_like(c_array)
        c_array[0] = 1
        c_array = expand_dims(c_array,axis=(1,2))
    ep_eff    = sqrt(sum(ep_array*c_array,axis=0))
    return ep_eff

def GLAD(material):
    material_GLAD,ang = material["refindex"],material["deposition_angle"]
    n_eff = extract_refindex(material_GLAD,ang=ang)
    return n_eff



if __name__=="__main__":
    print("""
    This is an Effective Medium Approxmiation Module, developed by Incheol Jung.
    Should there be any errors or unpredicted results, please contact me.
    Email:  jungin1107@gmail.com
    """)