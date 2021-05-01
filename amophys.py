"""
    AMO/QM functions!
    
    Preston Huft, 2019
    
    Eventually bundle stuff into classes? idk, i'm usually fine importing everything
"""

#### libraries
from sympy.physics.wigner import wigner_6j,wigner_3j,clebsch_gordan
from sympy import symbols,N,sympify,lambdify
from sympy import MatrixSymbol,MatAdd,MatMul,Identity as eye,Matrix,zeros
from sympy.utilities.iterables import flatten
import numpy as np
from numpy import *
from numpy.linalg import eig
from numpy.random import normal
from scipy.optimize import curve_fit
from random import random as rand


#### local files
from physconsts import *
from rbconsts import *

#### Electric and Magnetic Dipole Interactions 

def gF_fn(F,J,I,gJ,gI):
    """ Returns the F lande g factor """

    return (gJ*(F*(F+1)+J*(J+1)-I*(I+1))
            +gI*(F*(F+1)-J*(J+1)+I*(I+1)))/(2*F*(F+1))

def gJ_fn(J,L,S,gL,gS):
    """ Returns the J lande g factor """

    return (gL*(J*(J+1)+L*(L+1)-S*(S+1)) 
            +gS*(J*(J+1)-L*(L+1)+S*(S+1)))/(2*J*(J+1))

def hf_zeeman(states,gJ,gI,Bz=None,units='Joules'):
    """ Return Zeeman Hamiltonian in hyperfine basis |L J I F mF>. Assumes the 
        field is along the z axis, i.e. q = 0.

        'states': list-like, pair of quantum states given by [I,J,F,mF,FF,mFF]
        If Bz is none initially, the sympy free_symbol is the magnetic dipole energy,
        uB*B, not just B. 
        
        'units': 'Joules' (default), 'eV', 'UB' (units of the magnetic dipole 
        energy).
        
        From Mark's notes for the general hf Zeeman matrix elements. Units
        are determined by UB. Could implement decorator function to change
        units.
    """
        
    ## TODO: build in better unit functionality
        
    I,J,F,mF,FF,mFF = states
    q = 0 # assume B = Bz for now

    elem = 0
    if mF == mFF:
        elem += N(clebsch_gordan(F,1,FF,mF,q,mFF) \
                *sqrt(2*F+1)*(-1)**(1+J+I) \
                *(gJ*(-1)**F*sqrt(J*(J+1)*(2*J+1)) \
                *wigner_6j(J,I,F,FF,1,J) \
                +gI*(-1)**FF*sqrt(I*(I+1)*(2*I+1)) \
                *wigner_6j(I,J,F,FF,1,I))) 
        # N() is used to ensure diagnolization doesn't get tripped up

    if Bz is not None:
        elem *= uB*Bz # hmm check the sign
        if units == 'Joules':
            return elem
        elif units == 'eV':
            return JToeV(elem)
        elif units == 'GHz':
            return eVToGHz(JToeV(elem)) 
        else:
            print(f"invalid unit [{units}] for non-zero Bz. Result in 'J'.")
            return elem
    else: 
        if units == 'UB':
            return elem
        else:
            UB = symbols('U_B') # symbolic B field P.E. for now
            elem*= UB
            return elem
            
def hf_coupling(F,mF,J,q,FF,mFF,JJ,I,RME=None):
    """ 
    Returns the matrix element <F,mF,J|T_q|F',mF',J'>. 
    Args:
        'RME': the reduced matrix element <alpha;J||r||alpha'J'> with alpha 
        including quantum numbers not relevant to the coupling, e.g. n. 
        If RME=None, the matrix element is in units of [RME].
        
        I is the nuclear spin of the atom.
    Returns:
        'mat_elem': 
        ## From Mark's notes, eqs. A-50,51. Also see Steck Rb datasheet.
        mat_elem = rme*pow(-1,F+JJ+1+I)*sqrt((2*F+1)*(2*JJ+1)) \
                   *wigner_6j(J,I,F,FF,1,JJ) \
                   *clebsch_gordan(1,F,FF,q,mF,mFF)
    """

    rme = 1
    if RME!=None:
        rme = RME

    ## From Mark's notes, eqs. A-50,51
    mat_elem = rme*pow(-1,F+JJ+1+I)*sqrt((2*F+1)*(2*JJ+1)) \
                *wigner_6j(J,I,F,FF,1,JJ) \
                *clebsch_gordan(1,F,FF,q,mF,mFF)

    return mat_elem

def f_coupling(L,J,mJ,q,LL,JJ,mJJ,I,RME=None):
    """ 
    Returns the matrix element <J,mJ|T_q|J',mJ'>. 
    Args:
        'RME': the reduced matrix element <alpha;J||r||alpha'J'> with alpha 
        including quantum numbers not relevant to the coupling, e.g. n. 
        If RME=None, the matrix element is in units of [RME].
        
        I is the nuclear spin of the atom.
    Returns:
        'mat_elem': 
        ## From Mark's notes, eqs. A-50,51. Also see Steck Rb datasheet.
        mat_elem = rme*pow(-1,F+JJ+1+I)*sqrt((2*F+1)*(2*JJ+1)) \
                   *wigner_6j(J,I,F,FF,1,JJ) \
                   *clebsch_gordan(1,F,FF,q,mF,mFF)
    """

    rme = 1
    if RME!=None:
        rme = RME

    ## From Mark's notes, eqs. A-50,51
    mat_elem = rme*clebsch_gordan(1,J,JJ,q,mJ,mJJ)

    return mat_elem
    
def hamiltonian_z1(basis,gI,gJ,Bz=None,I=1.5,J=.5,units='Joules'):
    """ returns the hf zeeman hamiltonian Z1 for a provided basis.
        'Bz': the field strength [T]. None by default, then hamiltonian
        can be lambdified for the field energy UB = - muB*Bz. 
        
        'units': 'Joules','eV', more depending on matrix elem call 
        
        For now, assumes 87Rb (I = 3/2) ground states (J=1/2)
    """
    #TODO: make general. could specify a basis format by string, e.g. 
    # ['I', 'J', 'F', 'mF']

    dim = len(basis)
    H_Zz = np.empty((dim,dim),object)
        
    for i,state_i in enumerate(basis):
        F,mF = state_i
        for j,state_j in enumerate(basis):
            FF,mFF = state_j
            states = [I,J,F,mF,FF,mFF]
            try:
                H_Zz[i,j] = hf_zeeman(states,gJ,gI,Bz=Bz,units=units)
            except:
                print("Failed: %s" % states)
                print(gJ,gI,Bz)

    return H_Zz
    
def hamiltonian(basis,mat_elem):
    """ 
        returns a hamiltonian in the given basis, whose elements are to 
        be specified by a decorator function. 
    """
    #TODO: make general. could specify a basis format by string, e.g. 
    # ['I', 'J', 'F', 'mF']

    dim = len(basis)
    H = np.empty((dim,dim),object)
        
    for i,state_i in enumerate(basis):
        F,mF = state_i
        for j,state_j in enumerate(basis):
            FF,mFF = state_j
            states = [I,J,F,mF,FF,mFF]
            try:
                H_Zz[i,j] = mat_elem(states)
            except:
                print("Failed: %s" % states)
                print(gJ,gI,Bz)

    return H
    
def acstark_twolevel(O,D):
    """ The two level atom AC Stark shift from diagnolization of the full Hamiltonian,
        assuming the detuning D << E2-E1, and making the RWA. 
        
        Uac = -O**2/(4*D)
    """
    
    return -O**2/(4*D)

    if RME is None:
        RME = 1 # the light shift is now in units of the RME

    return -(ee**2)*w_ab*cc(RME)*RME*I/(2*hbar*(w_ab**2-w**2))
    
def rabi(I0,matelem):
    """ the rabi frequency for a beam of intensity I0 coupling states |e>,|g>
        such that matelem = <e|q x|g>. May need to include a coupling const. 
        as prefactor on matelem. 
    """
    return sqrt(2*I0/(c*e0))*matelem/hbar
    
#### Atomic State Evolution

def obe_derivs(y0,t,D,O,phi=0,t1=np.inf,t2=np.inf):
# def derivs(t,y0,params):
    """ Returns RHS of optical bloch eqs for current values at time t,
    
        drgg = ree/t1 - 1j/2*(O*cc(reg)-cc(O)*reg) 
        dree = -ree/t1 + 1j/2*(O*cc(reg)-cc(O)*reg)
        dreg = (1j*D-1/(2*t1))*reg+1j*O/2*(rgg-ree) # = cc(drge)
    
        'y0': [rgg,ree,reg]
        't': time
        'D': detuning
        'O': Rabi frequency
        't1': state lifetime
        't2': coherence
    """
    
    rgg,ree,reg = y0
    
    # time derivatives of density op elements
    curl = 1j/2*(O*cc(reg)-cc(O)*reg) 
    drgg = ree/t1 - curl 
    dree = -ree/t1 + curl
    dreg = (1j*D-1/(2*t1))*reg+1j*O/2*(rgg-ree) # actually reg tilda
    
    return array([drgg,dree,dreg])

#### Various classes

class dipole_trap:
    
    def __init__(self,lmbda,wx,Tdepth,Tatom,wy=None):
        """ A dipole trap object with the beams potential and distribution of 
            atoms specified by Tatom. 
            'wx': x beam waist in focal plane (z=0)
            'wy': Assumed equal to wx by default
            'Tdepth'
            'Tatom'
        """
        self.wx = wx
        self.Tdepth = Tdepth
        self.T = Tatom
        if wy is None:
            self.wy = wx
        else:
            self.wy = wy
        
        # FORT and atom parameter stuff
        self.umax = kB*self.Tdepth # the maximum FORT depth
        self.lmbda = lmbda # the trap wavelength [m]

        self.zR = pi*wx**2/self.lmbda
        self.omega_r = (1/sqrt((self.wx**2+self.wy**2)/2))*sqrt(2*kB*self.Tdepth/mRb) # radial trap frequency 
        self.omega_z = (1/self.zR)*sqrt(2*kB*self.Tdepth/mRb) # axial trap frequency
        # print(f"omega_r = {self.omega_r*1e-3:.3f} kHz, omega_z = {self.omega_z*1e-3:.3f} kHz")

    def U(self,x,y,z):
        """ the potential energy as a function of space in the dipole trap.
            ASTIGMATISM not correctly represented!!
        """
        zR = self.zR
        wx = self.wx
        wy = self.wy
        ww = (1+z**2/zR**2) 
        umax = self.umax
        return -umax*exp(-2*x**2/(wx**2*ww)-2*y**2/(wy**2*ww))/ww 
    
    def xdist(self,events=None,plane=None):
        """ velocity component distributions """
        # Grainger group method
        omega_r = self.omega_r
        omega_z = self.omega_z
        T = self.T
        dx = dy = sqrt(kB*T/(mRb*omega_r**2))
        dz = sqrt(kB*T/(mRb*omega_z**2))
        zlist = normal(0,dz,size=events)
        xlist = normal(0,dx,size=events)
        ylist = normal(0,dy,size=events)
        
        if plane is 'xz':
            return xlist,zlist
        else:
            if events is None:
                return xlist[0],ylist[0],zlist[0]
            return xlist,ylist,zlist
    
    def vdist(self,events):
        """ maxwell boltzmann speeds """
        umax = self.umax
        atoms = ensemble(self.T)
        
        vlist = atoms.sampling_maxboltzv(events,[0,2]) # speeds

        vxlist = empty(events)
        vylist = empty(events)
        vzlist = empty(events)
        
        for i in range(events):
            ex = 2*rand()-1
            ey = 2*rand()-1
            ez = 2*rand()-1
            v = vlist[i]
            A = sqrt(ex**2+ey**2+ez**2)
            vxlist[i] = ex*v/A
            vylist[i] = ey*v/A
            vzlist[i] = ez*v/A
            
#         vlist = array([sqrt(vx**2+vy**2+vx**2) for vx,vy,vz in zip(vxlist,vylist,vzlist)])
#         plt.hist(vlist, 50, density=True) # show maxwell boltzmann speed dist
#         plt.xlabel('v')
#         plt.ylabel('occurences')
#         plt.show()

        return vxlist,vylist,vzlist
    
    def distplot(self,events):
        """ show atoms in FORT in z = 0 plane before drop and recapture """
        mu = 1e-6
        wx = self.wx
        zR = self.zR
        print(f"zr={zR/mu:.0f} [um], wx={wx/mu:.0f} [um]")
        
        xlist,ylist = self.xdist(events,plane='xz') # positions in [m]
        
        s = 1.5 # units waist or rayleigh length
        xpts = linspace(-s*wx,s*wx,100)
        zpts = linspace(-s*zR,s*zR,100)
        xx,zz = meshgrid(xpts,zpts)
        fpts = -self.U(xx,0,zz) # the fort intensity eval'd on the meshgrid
        plt.contourf(xpts/mu,zpts/mu,fpts)
        plt.scatter(xlist/mu,ylist/mu,color='red')
        plt.xlabel("x")
        plt.ylabel("z")
#         plt.axes().set_aspect('equal')
        plt.show() 

    def drop_recap(self,tlist,T=None,events=None,base_retention=None,
            progress=False):
        """ Procedure for simulating a release ("drop") and recapture experiment
            to deduce the temperature of actual atoms in such an experiment. 
        
            Based on code by Mark, with some corrections
            'wx': waist
            'Tdepth': FORT temperature depth
            'T': atom temp
            'tmax': max time in units us
            'steps': number of FORT drop outs
            'events': number of release-recapture events per data pt
            'wy': optional waist for eliptical FORT 
        """
      
        Tdepth = self.Tdepth
        wx = self.wx
        umax = self.umax
        zR = self.zR
        tlist = 1e-6*tlist

        if T is None:
            T = self.T
        if events is None:
            events = 2000
        if base_retention is None:
            base_retention = 1 # the retention baseline with no fort drop

        retention = empty(len(tlist))
   
        xlist,ylist,zlist = self.xdist(events)
        vzlist,vxlist,vylist = self.vdist(events)

        for j,t in enumerate(tlist):

            escape = 0 
            nhot = 0 # this is an untrapped atom

            for i in range(events):     
                hot = 0
                KE = .5*mRb*((vxlist[i]-g*t)**2+vylist[i]**2
                              +vzlist[i]**2)
                PE0 = self.U(xlist[i],ylist[i],zlist[i])
                PE = self.U(xlist[i]+t*vxlist[i]+.5*g*(t)**2,
                       ylist[i]+t*vylist[i],
                       zlist[i]+t*vzlist[i])

                if KE + PE0 > 0:
                    hot = 1
                nhot += hot
                if KE + PE > 0:
                    escape += 1-hot
            retention[j] = base_retention*(1 - escape/events)
            
            if progress is not False:
                if j % 10 == 0:
                    print(f"timestep {j}: t = {t*1e6:.0f} [us], ret = {retention[j]:.2f}")     
        
        print(f"finished. T={T*1e6} [uK], r = {base_retention}")
        return tlist*1e6,retention
    
    def curvefit(self,tdata,rdata):
        """ For using the release_recap procedure to fit real data.
            
            tdata: time pts from data
            rdata: retention pts data
        """
        def f(tlist,T,r):
            """
                tlist: times [s]
                T: atom temp [K]
                r: baseline retention in (0,1]
            """
            
            t,y = self.drop_recap(tlist,T=T,base_retention=r)
            return y
        
        p0=[self.T,1] # scale up the temp 
        popt,pcov = curve_fit(f,tdata,rdata,p0,
                              absolute_sigma=True)
        
        Topt,ropt = popt # the optimum parameters
        
        print(f"T_fit = {Topt*1e6:.0f}")
        tlist,ret=self.drop_recap(tdata,T=Topt,base_retention=ropt)
        
        plt.plot(tlist,ret,label=f"T_fit = {Topt*1e6:.0f}")
        plt.scatter(tdata,rdata,label='real live data',color='r')
        plt.show()
        
        return tlist,ret

#### Optics

class gaussian_beam:
    
    ## TODO: figure out how to have methods not tied to object ref, 
    ## which can be called individually, and also ref each other
    
    @classmethod
    def intensity(x,y,z,lmbda,wx,I0,wy=None,z_offset=0):
        if wy is None:
            wy = wx
            
        zRx = z_rayleigh(lmbda,wx)
        zRy = z_rayleigh(lmbda,wy)

        wzx = (1+(z/zRx)**2)
        wzy = (1+(z/zRy)**2)
        return I0*exp(-2*x**2/(wzx*(wx**2*wzx))-2*y**2/(wzy*(wy**2*wzy)))	
    
    
def z_rayleigh(lmbda,w0):
        return pi*w0**2/lmbda
        
#### Quantum Physics

def jmbasis(jlist):
    """ returns a numpy array of basis vectors {|J,mJ>} given a list of 
        J vals. Output is flipped w.r.t. order of j's passed in.		
    """
    
    # TODO: make flip optional
    basis = np.empty(sum([2*j+1 for j in jlist]),list)
    i = 0
    for j in jlist:
        for m in range(-j,j+1):
            basis[i] = [j,m]
            i+=1 
    return np.flip(basis)

def comm(A,B):
    """ Returns the commutator of A,B: [A,B]=A.B-B.A. Assumes 'A','B' are sympy 
        matrices."""
        
        # TODO: extend to check type, and work for numpy matrices/ndarrays
    return (MatMul(A,B)-MatMul(B,A))

#### Syntactic sugar and convenience functions

def czeros(num): # numpy can already do this with zeros(num,complex)
    """ return array of complex zeros """
    c = empty(num,complex)
    for i in range(num):
        c[i] = 0+0j
    return c

def cc(z):
    """ return numpy.conj(z)"""
    return conj(z)
    
def diagonal(mat):
    """ 
        diagonalize mat, but with eigenvalues ordered from largest (0,0) to
        smallest (N,N) for mat NxN. Warning: this casts the output to a np 
        array for now. 
        
        'mat' assumed to be sympy matrix. 
    """
    # TODO: extend to support numpy square ndarray diagnolization as well
    assert shape(mat)[0]==shape(mat)[1], "the matrix is not square"
    dim = shape(mat)[0]
    P,D = mat.diagonalize()
    D_copy = copy(D) # guarantees the return type the same as input type
    
    try:
        eigs = flip(sort([D[i,i] for i in range(dim)]))
        for i in range(dim):
            D_copy[i,i] = eigs[i]
        return Matrix(D_copy)
    except:
        print("make sure that the matrix is numeric")
        return D


