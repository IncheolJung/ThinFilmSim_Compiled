import numpy as np
# import cupy as cp
import pandas as pd
import csv, os
from scipy.interpolate import interp1d,interp2d

from EffectiveMediumApproximation import *

import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)



class PolarizationError(Exception):
    def __str__(self):
        return "Polarization not defined correctly"



def interpolate(wvl,datapath1=[".\\Data",],output=".\\Data_converted",
                ColorMatching_datapath=False,GLAD_datapath=False,info=None):
    
    x_interp = wvl

    if info: info.update(f"Cleaning Old Data....")
    A = os.listdir(output)
    for file in A: os.remove(output+"\\"+file)
    
    if type(datapath1)==list: datapaths = datapath1
    else: datapaths = [datapath1]
    if GLAD_datapath: datapaths.append(GLAD_datapath)
    
    if ColorMatching_datapath:
        if info: info.update(f"Interpolating Color Matching Functions....")
        Cfnc = np.array(pd.read_excel(ColorMatching_datapath+"\\color_matching_function.xlsx"))
        D65 = np.array(pd.read_excel(ColorMatching_datapath+"\\D65.xlsx"))
        AM15 = np.array(pd.read_excel(ColorMatching_datapath+"\\AM15.xlsx"))
        wvl_C,x,y,z = Cfnc[:,0],Cfnc[:,1],Cfnc[:,2],Cfnc[:,3]
        wvl_D,lumin = D65[:,0],D65[:,1]
        wvl_A,illum = AM15[:,0],AM15[:,1]
        fqx = interp1d(wvl_C,x,kind="quadratic",bounds_error=False,fill_value=0)
        fqy = interp1d(wvl_C,y,kind="quadratic",bounds_error=False,fill_value=0)
        fqz = interp1d(wvl_C,z,kind="quadratic",bounds_error=False,fill_value=0)
        fqD = interp1d(wvl_D,lumin,kind="quadratic",bounds_error=False,fill_value=0)
        fqA = interp1d(wvl_A,illum,kind="quadratic",bounds_error=False,fill_value=0)
        with open(ColorMatching_datapath+"_converted\\color_matching_function.csv",'w',newline='',encoding='utf-8') as f:
            wr = csv.writer(f)
            temp = np.array([x_interp,fqx(x_interp),fqy(x_interp),fqz(x_interp)]).T
            for ii in temp:
                wr.writerow(ii)
        with open(ColorMatching_datapath+"_converted\\D65.csv",'w',newline='',encoding='utf-8') as f:
            wr = csv.writer(f)
            temp = np.array([x_interp,fqD(x_interp)]).T
            for ii in temp:
                wr.writerow(ii)
        with open(ColorMatching_datapath+"_converted\\AM15.csv",'w',newline='',encoding='utf-8') as f:
            wr = csv.writer(f)
            temp = np.array([x_interp,fqA(x_interp)]).T
            for ii in temp:
                wr.writerow(ii)
    for datapath in datapaths:
        B = os.listdir(datapath)
        refdata = []
        for ii,filename in enumerate(B):
            material_name = filename.split(".")[0]
            if info: info.update(f"Reading data....{material_name}")
            temp = pd.read_excel(datapath+"\\"+filename,header=None,sheet_name=None)
            # try:
            #     temp = pd.read_excel(datapath+"\\"+filename,header=None,dtype=float,sheet_name=None)
            # except ValueError:
            #     temp = pd.read_excel(datapath+"\\"+filename,header=None)
            #     a = temp.dropna(axis=1,how="all").values
            #     for threshold,e in enumerate(a):
            #         try: alpha = e.astype(float)
            #         except ValueError: alpha=np.nan
            #         if np.isfinite(alpha).all(): break
            #     temp = pd.read_excel(datapath+"\\"+filename,header=None,dtype=float,skiprows=threshold,sheet_name=None)
            data = []
            for ii,(key,value) in enumerate(temp.items()):
                value = value.dropna(axis=0,how="all").dropna(axis=1,how="all")
                threshold = 0
                for value_row in value.values:
                    try:
                        float(value_row[0])
                        break
                    except ValueError: threshold+=1
                data_angle = np.array(value[threshold:],dtype=float)
                try:
                    angle = float(key.split(" ")[0])
                except ValueError:
                    angle = 0
                    data_angle = np.hstack([data_angle[:,:3],data_angle[:,1:3]])
                # try:
                #     angle = float(key.split(" ")[0])
                #     data_angle = np.array(value.dropna(0,how="all").dropna(1,how="all"))
                # except:
                #     angle = 0
                #     data_angle = np.array(value.dropna(0,how="all").dropna(1,how="all"))
                #     data_angle = np.hstack([data_angle,data_angle[:,1:]])
                data.append([angle,data_angle])
            refdata.append([material_name,data])
        for row in refdata:
            material_name,data = row
            if info: info.update(f"Converting data....{material_name}")
            y = [e[0] for e in data]
            znp,zkp,zns,zks = [],[],[],[]
            for data_row in data:
                angle,data_angle = data_row
                x,data_np,data_kp,data_ns,data_ks = data_angle.T
                fqnp = interp1d(x,data_np,kind="quadratic",fill_value="extrapolate")
                fqkp = interp1d(x,data_kp,kind="quadratic",fill_value="extrapolate")
                fqns = interp1d(x,data_ns,kind="quadratic",fill_value="extrapolate")
                fqks = interp1d(x,data_ks,kind="quadratic",fill_value="extrapolate")
                znp.append(fqnp(x_interp)); zkp.append(fqkp(x_interp))
                zns.append(fqns(x_interp)); zks.append(fqks(x_interp))
            znp,zkp,zns,zks = np.array(znp),np.array(zkp),np.array(zns),np.array(zks)
            # print(f"{material_name}  |  {znp.shape}")
            if len(y)==1:
                y.append(89)
                ones,zeros = np.ones_like(znp),np.zeros_like(znp)
                znp,zkp = np.array([znp,ones]),np.array([zkp,zeros])
                zns,zks = np.array([zns,ones]),np.array([zks,zeros])
            fqnp = interp2d(x_interp,y,znp,kind="linear")
            fqkp = interp2d(x_interp,y,zkp,kind="linear")
            fqns = interp2d(x_interp,y,zns,kind="linear")
            fqks = interp2d(x_interp,y,zks,kind="linear")
            filename = f"{output}\\{material_name}.npy"
            # data_interp = np.array([np.tile(x_interp,(len(y_interp),1)),\
            #     fqnp(x_interp,y_interp),fqkp(x_interp,y_interp),fqns(x_interp,y_interp),fqks(x_interp,y_interp)])
            # data_interp = np.moveaxis(data_interp,0,-1)
            data_interp = [fqnp,fqkp,fqns,fqks,]
            np.save(filename,data_interp)
    if info: info.terminate()
    return



def importdata(datapath=".\\Data_converted"):
    B = os.listdir(datapath)
    data = []
    for filename in B:
        temp = [filename.split(".")[0]]
        temp.append(np.load(datapath+"\\"+filename,allow_pickle=True))
        data.append(temp)
    return data



def getAbsorption(TMMresults,structure,wvlths,Efields=None):
    # try: [ts,tp,bts,btp,rs,rp,brs,brp],ratio,getElements,getPropPS = TMMresults
    # except ValueError:
    #     index,[ts,tp,bts,btp,rs,rp,brs,brp],ratio,getElements,getPropPS = TMMresults
    index,[ts,tp,bts,btp,rs,rp,brs,brp],ratio,subCoef,getElements,getPropPS = TMMresults
    Absorbs,Absorbp = np.array([[]]),np.array([[]])
    if Efields: CurLayer = len(Efields[0])
    for ii,st in enumerate(reversed(structure)):
        if st[1] and st[5]:
            z = np.linspace(0,st[1],int(np.ceil(st[1])))
            cur_phi,cur_refindex,cur_angle = getPropPS(-1-ii,z)
            inc_phi,inc_refindex,inc_angle = getPropPS(0,z)
            if Efields is None:
                a_s,a_p,b_s,b_p,c_s,c_p,d_s,d_p = getElements(ii)
                Es = ts*a_s*np.exp(-1j*cur_phi)+ts*c_s*np.exp(+1j*cur_phi)
                Ep = tp*a_p*np.exp(-1j*cur_phi)+tp*c_p*np.exp(+1j*cur_phi)
                Ex = Ep*np.cos(cur_angle)
                Ey = Es
                Ez = Ep*np.sin(cur_angle)
            else:
                start,end = CurLayer-len(z),CurLayer
                Ex = Efields[0][start:end]
                Ey = Efields[1][start:end]
                Ez = Efields[2][start:end]
                CurLayer = start
            Fs = abs(Ey**2); Fp = abs(Ex**2)+abs(Ez**2)
            nn,kk = cur_refindex.real,-cur_refindex.imag
            no,co = inc_refindex.real,np.cos(inc_angle)
            AbCoef = (4*np.pi*nn*kk)/(no*co*wvlths)
            try:
                Absorbs = np.append(AbCoef*Fs,Absorbs,axis=0)
                Absorbp = np.append(AbCoef*Fp,Absorbs,axis=0)
            except ValueError:
                Absorbs = AbCoef*Fs
                Absorbp = AbCoef*Fp
            # Absorbs.insert(0,AbCoef*Fs); Absorbp.insert(0,AbCoef*Fp)
    return Absorbs,Absorbp



def getEfields(TMMresults,structure,wvlths):
    # try: [ts,tp,bts,btp,rs,rp,brs,brp],ratio,getElements,getPropPS = TMMresults
    # except ValueError:
    #     index,[ts,tp,bts,btp,rs,rp,brs,brp],ratio,getElements,getPropPS = TMMresults
    index,[ts,tp,bts,btp,rs,rp,brs,brp],ratio,subCoef,getElements,getPropPS = TMMresults
    Efieldx,Efieldy,Efieldz = [],[],[]
    for ii,st in enumerate(reversed(structure)):
        if st[1] and st[5]:
            z = np.linspace(0,st[1],int(np.ceil(st[1])))
            cur_phi,cur_refindex,cur_angle = getPropPS(-1-ii,z)
            inc_phi,inc_refindex,inc_angle = getPropPS(0,z)
            a_s,a_p,b_s,b_p,c_s,c_p,d_s,d_p = getElements(ii)
            Es = ts*a_s*np.exp(-1j*cur_phi)+ts*c_s*np.exp(+1j*cur_phi)
            Ep = tp*a_p*np.exp(-1j*cur_phi)+tp*c_p*np.exp(+1j*cur_phi)
            Ex = Ep*np.cos(cur_angle)
            Ey = Es
            Ez = Ep*np.sin(cur_angle)
            Efieldx.extend(reversed(Ex))
            Efieldy.extend(reversed(Ey))
            Efieldz.extend(reversed(Ez))
    Efieldx.reverse(); Efieldy.reverse(); Efieldz.reverse()
    EfieldsProfile = [np.array(Efieldx),np.array(Efieldy),np.array(Efieldz),]
    return EfieldsProfile



def getAdmittance(TMMresults,structure):
    index,Coef,ratio,subCoef,getElements,getPropPS = TMMresults
    Admittance_real,Admittance_imag = [],[]
    for ii,st in enumerate(reversed(structure)):
        if st[1] and st[5]:
            curN = getPropPS(-1-ii,[0])[1][index]
            radius = np.max([10,np.abs((curN**2-oldN**2)/(2*oldN**2))])
            mod = np.abs(curN-oldN)
            # print(f"{st[0]}: {radius}")
            # print(f"{st[0]}: {mod}")
            num_z = np.max([4,int(np.ceil(mod*radius*5))])
            z = np.linspace(0,st[1],num_z)
            print(f"{st[0]}: {z.size}")
            cur_phi,cur_refindex,cur_angle = getPropPS(-1-ii,z)
            inc_phi,inc_refindex,inc_angle = getPropPS(0,z)
            curN,delta = cur_refindex[index],np.array(cur_phi.T[index])
            delta = interp1d(z,delta,kind="cubic")
            z_ = np.linspace(z.min(),z.max(),int(np.clip(100*z.size*3,0,1e6)))
            delta_ = delta(z_)
            sin,cos = np.sin(delta_),np.cos(delta_)
            admittance = (oldN*cos+1j*curN*sin)/(cos+1j*(oldN/curN)*sin)
            newx,newy = admittance.real,admittance.imag
            Admittance_real.append(newx); Admittance_imag.append(newy)
            oldN = newx[-1] + 1j*newy[-1]
        else:
            z = [0]
            old_phi,old_refindex,old_angle = getPropPS(-1-ii,z)
            oldN = old_refindex[index]
    Admittance = [Admittance_real,Admittance_imag]
    return Admittance



def Fresnel(n1,n2,th1=0,th2=0,pol=None):
    # pol: polarization (s/p)
    # th1,th2 = np.conj(th1),np.conj(th2)
    c1,c2 = np.cos(th1),np.cos(th2)
    if pol=="p":
        t = (2*n1*c1)/(n1*c2+n2*c1)
        r = (n1*c2-n2*c1)/(n1*c2+n2*c1)
    elif pol=="s":
        t = (2*n1*c1)/(n1*c1+n2*c2)
        r = (n1*c1-n2*c2)/(n1*c1+n2*c2)
    else: raise PolarizationError
    return t,r



def TMM(
    structure,refdata,wavelength,
    angle0=0,wvl0=None,
    Functions=False,get_refindex=False
    ):
    
    global overflow
    
    def getCoefs(N=-1,Ms=None,Mp=None,):
        ones,zeros = np.ones(len(wavelength)),np.zeros(len(wavelength))
        Ea_s,Ea_p,Eb_s,Eb_p,Ec_s,Ec_p,Ed_s,Ed_p = getElements(N,Ms,Mp)
        try: ts = 1/Ea_s
        except: ts = zeros; overflow = True
        try: tp = 1/Ea_p
        except: tp = zeros; overflow = True
        try: bts = (Ea_s*Ed_s-Eb_s*Ec_s)/Ea_s
        except: bts = zeros; overflow = True
        try: btp = (Ea_p*Ed_p-Eb_p*Ec_p)/Ea_p
        except: btp = zeros; overflow = True
        try: rs = Ec_s/Ea_s
        except: rs = ones; overflow = True
        try: rp = Ec_p/Ea_p
        except: rp = ones; overflow = True
        try: brs = -Eb_s/Ea_s
        except: brs = ones; overflow = True
        try: brp = -Eb_p/Ea_p
        except: brp = ones; overflow = True
        return ts,tp,bts,btp,rs,rp,brs,brp
    
    def getElements(N=-1,Ms=None,Mp=None,):
        if Ms is None: Ms = TransferMat_s
        if Mp is None: Mp = TransferMat_p
        [Ea_s,Eb_s],[Ec_s,Ed_s] = np.moveaxis(Ms[N],0,-1)
        [Ea_p,Eb_p],[Ec_p,Ed_p] = np.moveaxis(Mp[N],0,-1)
        return Ea_s,Ea_p,Eb_s,Eb_p,Ec_s,Ec_p,Ed_s,Ed_p
    
    def getPropPS(jj,zzz):
        cur_refindex_p = refindex_p[jj]
        cur_angle_p = angle_p[jj]
        propCoef = 2*np.pi*cur_refindex_p*np.cos(cur_angle_p)/wavelength
        propCoef_ = np.tile(propCoef,(len(zzz),1))
        zzz_ = np.tile(zzz,(len_wavelength,1)).swapaxes(0,1)
        Kz = propCoef_ * zzz_
        return Kz,cur_refindex_p,cur_angle_p
    
    np.seterr(all="raise",under="ignore")
    overflow = False
    len_wavelength = len(wavelength)
    ones,zeros = np.ones(len_wavelength),np.zeros(len_wavelength)
    EPSILON = 1e-6*ones
    # EPSILON_comp = EPSILON - 1j*EPSILON
    thickness,coh = [],[]
    refindex_p,refindex_s,phi_p,phi_s = [],[],[],[]
    angle_s,angle_p = [],[]
    for ii,layer in enumerate(structure):
        ref_p,ref_s = extract_refindex(layer[0],refdata,wavelength,ang=layer[3])
        refindex_p.append(ref_p); refindex_s.append(ref_s)
        thickness.append(layer[1])
        if ii:
            # ref_ratio_p = (refindex_p[-2]+EPSILON_comp)/(refindex_p[-1]+EPSILON_comp)
            # ref_ratio_s = (refindex_s[-2]+EPSILON_comp)/(refindex_s[-1]+EPSILON_comp)
            ref_ratio_p = refindex_p[-2]/refindex_p[-1]
            ref_ratio_s = refindex_s[-2]/refindex_s[-1]
            before_arcsin_p = ref_ratio_p*np.sin(angle_p[-1])
            before_arcsin_s = ref_ratio_s*np.sin(angle_s[-1])
            before_arcsin_p = np.where(
                before_arcsin_p.imag<0,
                before_arcsin_p.conjugate(),
                before_arcsin_p,
            )
            before_arcsin_s = np.where(
                before_arcsin_s.imag<0,
                before_arcsin_s.conjugate(),
                before_arcsin_s,
            )
            newangle_p = np.arcsin(before_arcsin_p)
            newangle_s = np.arcsin(before_arcsin_s)
            # from matplotlib.pyplot import subplots
            # hf,hx = subplots()
            # hx.set_title(layer[0])
            # hx.plot(wavelength,angle_p[-1].real,"g-",label=r"$\theta$")
            # hx.plot(wavelength,angle_p[-1].imag,"g:")
            # hx.plot(wavelength,before_arcsin_p.real,"r-",label=r"$\frac{n2}{n1}sin(\theta)$")
            # hx.plot(wavelength,before_arcsin_p.imag,"r:")
            # hx.plot(wavelength,newangle_p.real,"b-",label="prop_angle")
            # hx.plot(wavelength,newangle_p.imag,"b:")
            # hf.legend()
            # hf.show()
        else:
            newangle_p = np.tile(angle0,len_wavelength)
            newangle_s = np.tile(angle0,len_wavelength)
        angle_p.append(newangle_p)
        angle_s.append(newangle_s)
        phi_p.append(2*np.pi*thickness[-1]*refindex_p[-1]*np.cos(newangle_p)/wavelength)
        phi_s.append(2*np.pi*thickness[-1]*refindex_s[-1]*np.cos(newangle_s)/wavelength)
        coh.append(np.tile(layer[5],len_wavelength))
    coh[0],coh[-1] = ones,ones
    thickness[0],thickness[1] = True,True

    max_num = np.tile(709.78,len(wavelength))
    sTrMat,pTrMat,pPrMat,sPrMat,pSbMat,sSbMat = [],[],[],[],[],[]
    parameters = [
        reversed(refindex_p),reversed(refindex_s),
        reversed(phi_p),reversed(phi_s),
        reversed(angle_p),reversed(angle_s),
        reversed(thickness),reversed(coh)
        ]
    for ii,(nn_p,nn_s,pp_p,pp_s,th_p,th_s,dd,cc) in enumerate(zip(*parameters)):
        if ii:
            sb = 1-cc
            ts,rs = Fresnel(nn_s,newnn_s,th_s,newth_s,pol="s")
            try: exp_s = np.exp(+1j*pp_s.real*cc-pp_s.imag)
            except: exp_s = np.exp(+1j*pp_s.real*cc-max_num); overflow = True
            try: exm_s = np.exp(-1j*pp_s.real*cc+pp_s.imag)
            except: exm_s = np.exp(-1j*pp_s.real*cc+max_num); overflow = True
            stransitionM = np.moveaxis(np.array([[ones,cc*rs],[cc*rs,ones]])/ts,-1,0)
            spropagationM = np.moveaxis(np.array([[exp_s,zeros],[zeros,exm_s]]),-1,0)
            ssubstrateM = np.moveaxis(np.array([[exp_s**sb,exp_s*sb*rs],[exm_s*sb*rs,exm_s**sb]]),-1,0)
            sTrMat.append(stransitionM)
            sPrMat.append(spropagationM)
            sSbMat.append(ssubstrateM)
            tp,rp = Fresnel(nn_p,newnn_p,th_p,newth_p,pol="p")
            try: exp_p = np.exp(+1j*pp_p.real*cc-pp_p.imag)
            except: exp_p = np.exp(+1j*pp_p.real*cc-max_num); overflow = True
            try: exm_p = np.exp(-1j*pp_p.real*cc+pp_p.imag)
            except: exm_p = np.exp(-1j*pp_p.real*cc+max_num); overflow = True
            ptransitionM = np.moveaxis(np.array([[ones,cc*rp],[cc*rp,ones]])/tp,-1,0)
            ppropagationM = np.moveaxis(np.array([[exp_p,zeros],[zeros,exm_p]]),-1,0)
            psubstrateM = np.moveaxis(np.array([[exp_p**sb,exp_p*sb*rp],[exm_p*sb*rp,exm_p**sb]]),-1,0)
            pTrMat.append(ptransitionM)
            pPrMat.append(ppropagationM)
            pSbMat.append(psubstrateM)
        newnn_p,newnn_s,newth_p,newth_s = nn_p,nn_s,th_p,th_s
    
    TransferMat_s  = np.tile(np.eye(2),(1,len_wavelength,1,1))
    TransferMat_p  = np.tile(np.eye(2),(1,len_wavelength,1,1))
    SubstrateMat_s = np.tile(np.eye(2),(1,len_wavelength,1,1))
    SubstrateMat_p = np.tile(np.eye(2),(1,len_wavelength,1,1))
    for sTr,pTr,pPr,sPr,pSb,sSb in zip(sTrMat,pTrMat,pPrMat,sPrMat,pSbMat,sSbMat):
        try: TransferMat_s_row = np.matmul(sTr,TransferMat_s[-1])
        except FloatingPointError: overflow = True
        try: TransferMat_s_row = np.matmul(sPr,TransferMat_s_row)
        except FloatingPointError: overflow = True
        try: TransferMat_p_row = np.matmul(pTr,TransferMat_p[-1])
        except FloatingPointError: overflow = True
        try: TransferMat_p_row = np.matmul(pPr,TransferMat_p_row)
        except FloatingPointError: overflow = True
        try: SubstrateMat_s_row = np.matmul(sSb,SubstrateMat_s[-1])
        except FloatingPointError: overflow = True
        try: SubstrateMat_p_row = np.matmul(pSb,SubstrateMat_p[-1])
        except FloatingPointError: overflow = True
        TransferMat_s = np.append(TransferMat_s,np.expand_dims(TransferMat_s_row,0),axis=0)
        TransferMat_p = np.append(TransferMat_p,np.expand_dims(TransferMat_p_row,0),axis=0)
        SubstrateMat_s = np.append(SubstrateMat_s,np.expand_dims(SubstrateMat_s_row,0),axis=0)
        SubstrateMat_p = np.append(SubstrateMat_p,np.expand_dims(SubstrateMat_p_row,0),axis=0)
    
    ts,tp,bts,btp,rs,rp,brs,brp = getCoefs(-1,TransferMat_s,TransferMat_p)
    if overflow:  Sts,Stp,Strs,Strp,Srs,Srp,Sbrs,Sbrp = [*[zeros]*4,*[ones]*4,]
    else: Sts,Stp,Strs,Strp,Srs,Srp,Sbrs,Sbrp = getCoefs(-1,SubstrateMat_s,SubstrateMat_p)
    # n_ini,n_fin = refindex_p[0],refindex_p[-1]
    # cos_ini,cos_fin = np.cos(angle_p[0]),np.cos(angle_p[-1])
    # ratio_s = n_fin*cos_fin/n_ini*cos_ini
    # ratio_p = n_fin.conjugate()*cos_fin/n_ini.conjugate()*cos_ini
    n_ini,n_fin = refindex_s[0],refindex_s[-1]
    cos_ini,cos_fin = np.cos(angle_s[0]),np.cos(angle_s[-1])
    ratio_s = np.real(n_fin*cos_fin)/(n_ini*cos_ini)
    n_ini,n_fin = refindex_p[0],refindex_p[-1]
    cos_ini,cos_fin = np.cos(angle_p[0]),np.cos(angle_p[-1])
    ratio_p = np.real(np.conj(n_fin)/cos_fin)/(np.conj(n_ini)/cos_ini)
    
    # if overflow: print("OVERFLOW!")
    res = []
    if wvl0:
        ii = np.argmin(np.abs(wavelength-wvl0))
        res.append(ii)
    res.append([ts,tp,bts,btp,rs,rp,brs,brp])
    res.append([ratio_s.real,ratio_p.real,])
    res.append([Sts,Stp,Strs,Strp,Srs,Srp,Sbrs,Sbrp,])
    if Functions:
        res.append(getElements)
        res.append(getPropPS)
    if get_refindex:
        res.append([refindex_s,refindex_p,])
    
    np.seterr(all="raise")

    return res



if __name__=="__main__":
    print("""
    This is a Transfer Matrix Module, developed by Incheol Jung.
    Should there be any errors or unpredicted results, please contact me.
    Email:  jungin1107@gmail.com
    """)