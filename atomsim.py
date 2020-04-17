#### modules
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from collections import OrderedDict
from scipy.integrate import solve_ivp
from sympy import MatrixSymbol,MatMul,I,Matrix,symbols,Function
from sympy.utilities.lambdify import lambdify
from physconsts import hbar

#### functions

def comm(A,B):
    """
    return commutator(A, B)[A,B]=A.B-B.A.
    'A', 'B': sympy matrices of the same shape
    """
    assert A.shape == B.shape, (f"A,B must have the same shape, but"+
                                f"A.shape={A.shape} != B.shape={B.shape}")
    return (MatMul(A,B)-MatMul(B,A)) #.as_mutable
    

# TODO: simplify the eqs once they have been generated. Floor things to zero?
# TODO: properly handle time-dependence; i.e. if t is a free symbol in 
# hamiltonian, then include last in the lambdifygenerated function args
# TODO: handle Louivillian op for handling decay
# TODO: could make this a classmethod in AtomSim
# TODO: could return eqs
def build_derivs(hamiltonian, decay=None, showeqs=False, lambdifyhelp=False):
    """
    return derivatives for RHS of von Neumann eq given a hamiltonian
    
    Args:
        'hamiltonian': a sympy Matrix of dimension N x N representing the full
            system Hamiltonian
            
        'lambdifyhelp': bool. if True, shows the docstring for the
            function generated by sympy Lambdify for the derivatives function,
            which is wrapped by the return function.
        
        'decay': (optional, None by default) a sympy Matrix of dimension N x N 
            representing the "Louivillian" or decay operator
            
    Returns:
        'derivs': a function with call signature derivs(t, y), 
            where t is time and y = [r00, r01, .. r11, r12, ..] is a list of 
            unraveled density matrix elements of type complex
    """
    
    assert hamiltonian.shape[0] == hamiltonian.shape[1], ("Hamiltonian"+
                                             " must be square, "+
                                             f"not of shape "+
                                              str(hamiltonian.shape))
    dims = hamiltonian.shape[0]
    # build a symbolic density matrix
    r = MatrixSymbol('r', dims, dims).as_mutable()
    for i in range(dims):
        for j in range(dims):
            r[i,j] = symbols(f'r{i}{j}')
            if i > j:
                r[i,j] = np.conj(r[j,i])
#     print(r) 

    # calculate [r, H]:
    rhs = -1j*comm(r, hamiltonian)/hbar

    # decay term if supplied:
    if decay != None:
        rhs -= MatMul(decay, r)/hbar

    # prune off non-redundant elements
    pruned_rhs = []
    for i in range(dims):
        for j in range(dims): 
            if i <= j:
                pruned_rhs.append(rhs[i,j])

    # sort arguments in the order of pruned_rhs
    args = list(rhs.free_symbols)
    args.sort(key=lambda x: x.__repr__())
    #args.append(symbols('t'))

    rhs = pruned_rhs

    if showeqs is True:
        print("Equations: \n")
        for var,eq in zip(args,rhs):
            print('D[' + str(var) + '] = ' + str(eq) + '\n')

    f = lambdify(args, rhs)

    if lambdifyhelp:
        print(help(f))

    # TODO make below general; maybe include the name of an additional args as
    # a kwarg and then check by name if it is in args
    
    # check whether t is in the args
    if 't' in [a.__repr__() for a in args]:    
        def derivs(t, y): 
            """
            the RHS of the von Neumann equation for a given hamiltonian
            
            this function is a lambdifygenerated function which returns the
            derivative side of D[rho] = -i*[r, H]/hbar where H is the
            hamiltonian specified when build_derivs was called
            
            Args: 
                't': (float) the current time in the simulation
                'y': (list-like) the unraveled non-redunant elements of the
                    density matrix, of type complex. i.e. for H 3 x 3, 
                    y = [r00, r01, r02, r11, r12, r33]
                    
            Returns:
                'D[y]': (list-like) the element-wise derivative of y given by
                    the generated equations. 
            """
            
            return f(*y,t)
    else:
        def derivs(t, y):  
            """
            the RHS of the von Neumann equation for a given hamiltonian
            
            this function is a lambdifygenerated function which returns the
            derivative side of D[rho] = -i*[r, H]/hbar where H is the
            hamiltonian specified when build_derivs was called
            
            Args: 
                't': (float) the current time in the simulation
                'y': (list-like) the unraveled non-redunant elements of the
                    density matrix, of type complex. i.e. for H 3 x 3, 
                    y = [r00, r01, r02, r11, r12, r33]
                    
            Returns:
                'D[y]': (list-like) the element-wise derivative of y given by
                    the generated equations. 
            """
            return f(*y)
            
    return derivs


#### the simulation wrapper class; pretties up solving and plotting simulations

class AtomSim:
    
    # TODO: update args
    def __init__(self, rho0, t_exp, dt=.01, tcentered=False, derivs=None,
                 hamiltonian=None, decay=None, fields=[]): 
        """
        Constructor for the
        The ODEs can be
        supplied explicitly by the 'derivs' function passed in or, if a
        'hamiltonian' sympy Matrix is supplied, the von Neumann ODE system will
        be generated.
        
        Args:
            'rho0': unraveled density matrix, non-redundant elements only
            'derivs': (function, optional) should return RHS elements of ODE
            't_exp': (float/int) experiment duration
            'dt': (float, optional) timestep
            'tcentered': (bool, optional) whether to shift the time array to be 
                centered at t=0
            'hamiltonian': (sympy Matrix, optional) the full Hamiltonian for
                the system to be solved.
            'decay': (sympy Matrix, optional) the Louivillian aka decay operator
                for the system. It will only be used if a Hamiltonian of the 
                same shape is supplied.
            'fields': (list of float/int or sympy expressions, optional) 
                nominally, the fields included in the Hamiltonian, which can be
                plotted with the option 'fields' in showplot() if supplied.
        """
        assert derivs or hamiltonian, ("Either derivs or hamiltonian must not"+
                                       "be none")
        
        self.rho0 = rho0
        self.rho = None # the solution
        self.t_exp = t_exp # duration
        self.dt = dt # timestep
        self.t = np.arange(self.t_exp,step=self.dt) # the timesteps
        self.tcentered = tcentered 
        if self.tcentered == True:
            self.t -= self.t_exp/2
        self.m = len(rho0) # number of non-redunant rho elements
        self.dim = int((-1 + np.sqrt(1 + 8*self.m))/2)
        self.idx_p, self.idx_c = self.rho_idx()

        self.populations = [] # values for each time step
        self.coherences = [] # values for each time step
        self.fields = fields # fields (lambda expressions) passed into derivs
        
        if hamiltonian != None:
            self.derivs = build_derivs(hamiltonian, decay=decay)
        else:
            self.derivs = derivs
            
        
    def runsim(self, t_exp=None, dt=0.01, tcentered=None):
        """ 
        call scipy solve_ivp to solve the system of equations

        Args:
            't_exp':
            'dt':
            'tcentered':
        
        Returns:
            return (rho, t), where rho is a list of solutions for each 
            non-redundant density matrix element each timestep and t is the
            list of timesteps
                
        """
        # (re) compute timesteps
        if dt != None or t_exp != None: 
            if t_exp != None:
                self.t_exp = t_exp
            if dt != None:
                self.dt = dt
            self.t = np.arange(self.t_exp,step=self.dt)
            if self.tcentered == True and tcentered is None:
                self.t -= self.t_exp/2
            
        # only try to center t_steps if not done previously
        if tcentered == True and self.t[0] == 0:
            self.tcentered = True
            self.t -= self.t_exp/2

        tspan = [self.t[0],self.t[-1]]
        soln = solve_ivp(self.derivs,tspan,self.rho0,t_eval=self.t)
        self.rho = np.array(soln.y)
        
        self.populations = np.array([self.rho[i] for i in self.idx_p])
        self.coherences = np.array([self.rho[i] for i in self.idx_c])
        
        return self.rho, self.t
    
    def plots(self, show=['populations', 'fields', 'mixing angle'], 
              loc='upper right', coherences=False, kwargs=None):
        """ 
        return list of Axes object(s) for the items in 'show'.
        At most 1 Axes object for 'populations' and/or 'coherences'.
        
        'coherences': True if population plot should show off-diagonal rho 
            elements. when population plot not shown, parameter does nothing
                
        """
        if self.rho is None:
            print('simulation hasn\'t been run yet')
            print('running simulation...')
            self.runsim()
        
        # could probably use kwargs here to allow passing in axes
        def plot_ax(ax, title): 
            ax.set_title(title)
            ax.set_xlim((self.t[0], self.t[-1]))
            return ax
        
        def pop_plot(ax, title):
            ax = plot_ax(ax, title)
            for n,p in enumerate(self.populations):
                ax.plot(self.t, np.real(p), label=rf'$\rho[{n},{n}]$')
            if coherences:
                for n,c in zip(self.idx_c, self.coherences):
                    # TODO: calculate m,n from self.idx_c
                    ax.plot(self.t, np.real(c))#, label=f'rho_{m,n}')
            return ax
                    
        def field_plot(ax, title):
            ax = plot_ax(ax, title)
            for i,f in enumerate(self.fields): # these are lambda functions
                if type(f) == sp.Mul: 
                    # TODO: redefine f as lambdified f
                    assert len(f.free_symbols) == 1, "multivariable fields not supported"
                    arg = list(f.free_symbols)[0]
                    f = lambdify(arg, f)
                ax.plot(self.t, [np.real(f(t)) for t in self.t], 
                        label=rf'$\Omega${i+1}')
            return ax 
                
        def mixing_plot(ax, title):
            ax = plot(ax, title)
            assert len(self.fields) == 2
            f1, f2 = self.fields 
            ax.plot(self.t, [np.atan(f2(t)/f1(t)) for t in self.t]) #TODO: check this
            return ax
            
        # plotdict = {plottype: propdict, ...}
        # want more plot types? add another dict entry and plot function
        plotdict = OrderedDict({'populations': 
                                {'show': False, 
                                 'title':'Density matrix elements',
                                 'plot_func': pop_plot},
                                'fields': 
                                {'show': False, 
                                 'title':'Applied fields',
                                 'plot_func': field_plot},
                                'mixing angle': 
                                {'show': False, 
                                 'title': 'State mixing angle',
                                 'plot_func': mixing_plot}
                               })

        for key in show:
            if key in plotdict:
                plotdict[key]['show'] = True
        
        fig, axes = plt.subplots(1, sum([plotdict[key]['show'] for key in 
                                 plotdict]), **kwargs)
        if type(axes) != np.ndarray: # only one subplot
            axes = [axes]

        ax_idx = 0
        for propdict in plotdict.values():
            if propdict['show']:
                axes[ax_idx] = propdict['plot_func'](axes[ax_idx], 
                                                     propdict['title'])
                axes[ax_idx].legend(loc=loc)
                ax_idx += 1

        return fig, axes
    
    def rho_idx(self): 
        """ 
        for a N x N density matrix unraveled into a list of 
        non-redundant elements, return two lists, one which 
        contains the indices of the population terms, and 
        the other the indices of the coherence terms. 
        """        
#         if N == self.dim:
#             m = self.m
#         else: 
#             m = int(N*(N + 1)/2)

        idx_p = [0]# population indices
        idx_c = [] # coherence indices

        j = 0
        last = 0
        for i in range(1,self.m): # could put this in a recursive function
            if i == last + self.dim - j:
                idx_p.append(i)
                last = i
                j += 1
            else:
                idx_c.append(i)

        return idx_p, idx_c