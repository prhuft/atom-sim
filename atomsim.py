#### modules
import matplotlib.pyplot as plt
from numpy import *
from scipy.integrate import solve_ivp

class AtomSim:
    """Atom internal dynamics simulation. 
       'rho0': unraveled density matrix, non-redundant elements only
       'derivs'
       't_exp'
    """
    
    def __init__(self, rho0, derivs, t_exp, dt=.01, tcentered=False): 
        self.rho0 = rho0
        self.rho = None # the solution
        self.derivs = derivs
        self.t_exp = t_exp # duration
        self.dt = dt # timestep
        self.t = arange(t_exp,step=self.dt) # the timesteps
        if tcentered == True:
            self.t -= self.t_exp/2
        self.m = len(rho0) # number of non-redunant rho elements
        self.dim = int((-1 + sqrt(1 + 8*self.m))/2)
        self.idx_p, self.idx_c = self.rho_idx()
        
        self.populations = [] # values for each time step
        self.coherences = [] # values for each time step
        
    def runsim(self, idx=0):
        """ returns rho, a list of solutions for each non-redundant density 
            matrix element each timestep. 
        """
        dt = 0.01 # timestep for DE solver
        tspan = [self.t[0],self.t[-1]]
        soln = solve_ivp(self.derivs,tspan,self.rho0,t_eval=self.t)
        self.rho = soln.y
        
        self.populations = [self.rho[i] for i in self.idx_p]
        self.coherences = [self.rho[i] for i in self.idx_c]
        
        return self.rho, self.t
    
    def plots(self, show=['populations, fields, coherences'], 
              loc='upper right'):
        """ return list of Axes object(s) for the items in 'show'.
            At most 1 Axes object for 'populations' and/or 'coherences'.
        """
        if self.rho is None:
            print('running simulation..')
            self.runsim()
        
        pop_str = 'populations'
        coh_str = 'coherences'
        field_str = 'fields'
        rho_plot = pop_str in show or coh_str in show
        field_plot = 'fields' in show
        fig, axes = plt.subplots(1, rho_plot + field_plot)
        if type(axes) != ndarray: # only one subplot
            axes = [axes]

        def population_ax(ax):
            """ for both population and coherence plots """
            ax.set_title("Density matrix elements")
            ax.set_xlim((self.t[0], self.t[-1]))
            return ax

        def fields_ax(ax):
            ax.set_title("Applied fields")
            ax.set_xlim((self.t[0], self.t[-1]))
            return ax

        i = 0
        if rho_plot:
            axes[i] = population_ax(axes[i])
            if pop_str in show:
                for n,p in enumerate(self.populations):
                    axes[i].plot(self.t, p, label=rf'$\rho[{n},{n}]$')
            if coh_str in show:
                for n,c in zip(self.idx_c, self.coherences):
                    # TODO: calculate m,n from self.idx_c
                    axes[i].plot(self.t, c)#, label=f'rho_{m,n}')
            i += 1
        if field_plot:
            axes[i] = fields_ax(axes[i])
        for ax in axes:
            ax.legend(loc=loc)
        return fig, axes
    
    def rho_idx(self): 
        """ for a N x N density matrix unraveled into a list of 
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