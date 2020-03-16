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
    
    def plots(self, show=['populations', 'fields', 'mixing angle'], 
              loc='upper right', coherences=False):
        """ return list of Axes object(s) for the items in 'show'.
            At most 1 Axes object for 'populations' and/or 'coherences'.
            'coherences': True if population plot should show off-diagonal rho elements.
                when population plot not shown, this parameter does nothing
        """
        if self.rho is None:
            print('running simulation..')
            self.runsim()
            
        def plot_ax(ax, title): # could probably use kwargs here 
            ax.set_title(title)
            ax.set_xlim((self.t[0], self.t[-1]))
            return ax
        
        def pop_plot(ax, title):
            ax = plot_ax(ax, title)
            for n,p in enumerate(self.populations):
                ax.plot(self.t, p, label=rf'$\rho[{n},{n}]$')
            if coherences:
                for n,c in zip(self.idx_c, self.coherences):
                    # TODO: calculate m,n from self.idx_c
                    axes[i].plot(self.t, c)#, label=f'rho_{m,n}')
                    
        def field_plot(ax, title):
            ax = plot(ax, title)
            for i,f in enumerate(self.fields): # these are lambda functions
                ax.plot(self.t, [f(t) for t in self.t], label=rf'$\Omega${i}')
                
        def mixing_plot(ax, title):
            ax = plot(ax, title)
            assert len(self.fields) == 2
            f1, f2 = self.fields 
            ax.plot(self.t, [atan(f2(t)/f1(t)) for t in self.t]) # double check this
            
        plotdict = {'populations': 
                    {'show': False, 
                     'title':'Density matrix elements',
                     'plot_func': pop_plot
                    },
                    'fields': 
                    {'show': False, 
                     'title':'Applied fields',
                     'plot_func': field_plot
                    }
                    'mixing angle': 
                    {'show': False, 
                     'title': 'State mixing angle',
                     'plot_func': mixing_plot
                    }
                   }

        for key in show:
            if key in plotdict
            plotdict[key]['show'] = True
        
        fig, axes = plt.subplots(1, sum([plotdict[key]['show'] for key in plotdict]))
        if type(axes) != ndarray: # only one subplot
            axes = [axes]

        i = 0
        if plotdict['populations']['show']:
            axes[i] = population_ax(axes[i])
            for n,p in enumerate(self.populations):
                axes[i].plot(self.t, p, label=rf'$\rho[{n},{n}]$')
            if coherences:
                for n,c in zip(self.idx_c, self.coherences):
                    # TODO: calculate m,n from self.idx_c
                    axes[i].plot(self.t, c)#, label=f'rho_{m,n}')
            i += 1
        if plotdict['fields']['show']:
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