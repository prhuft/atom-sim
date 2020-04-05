#### modules
import matplotlib.pyplot as plt
from collections import OrderedDict
from numpy import *
from scipy.integrate import solve_ivp

class AtomSim:
    
    def __init__(self, rho0, derivs, t_exp, dt=.01, tcentered=False,
                fields=[]): 
        """
        Atom internal dynamics simulation. Hence I did not use that acronym.
        This is essentially just a wrapper class for running the ODE solver
        with the additon of clean plotting functionality. 
        
        'rho0': unraveled density matrix, non-redundant elements only
        'derivs': should return RHS elements of ODE
        't_exp': experiment duration
        'dt': timestep
        'tcentered': boolean; whether to shift the time array to be centered at t=0
        """
        self.rho0 = rho0
        self.rho = None # the solution
        self.derivs = derivs
        self.t_exp = t_exp # duration
        self.dt = dt # timestep
        self.t = arange(self.t_exp,step=self.dt) # the timesteps
        if tcentered == True:
            self.t -= self.t_exp/2
        self.m = len(rho0) # number of non-redunant rho elements
        self.dim = int((-1 + sqrt(1 + 8*self.m))/2)
        self.idx_p, self.idx_c = self.rho_idx()

        self.populations = [] # values for each time step
        self.coherences = [] # values for each time step
        self.fields = fields # fields (lambda expressions) passed into derivs
        
    def runsim(self, t_exp=None, dt=None):
        """ 
        return (rho, t), where rho is a list of solutions for each 
        non-redundant density matrix element each timestep and t is the
        list of timesteps
            'idx
        """
        if dt != None or t_exp != None: 
            if t_exp != None:
                self.t_exp = t_exp
            if dt != None:
                self.dt = dt
            self.t = arange(self.t_exp,step=self.dt) # recompute timesteps

        tspan = [self.t[0],self.t[-1]]
        soln = solve_ivp(self.derivs,tspan,self.rho0,t_eval=self.t)
        self.rho = soln.y
        
        self.populations = [self.rho[i] for i in self.idx_p]
        self.coherences = [self.rho[i] for i in self.idx_c]
        
        return self.rho, self.t
    
    def plots(self, show=['populations', 'fields', 'mixing angle'], 
              loc='upper right', coherences=False, kwargs=None):
        """ 
        return list of Axes object(s) for the items in 'show'.
        At most 1 Axes object for 'populations' and/or 'coherences'.
        
        'coherences': True if population plot should show off-diagonal rho elements.
            when population plot not shown, this parameter does nothing
                
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
                ax.plot(self.t, real(p), label=rf'$\rho[{n},{n}]$')
            if coherences:
                for n,c in zip(self.idx_c, self.coherences):
                    # TODO: calculate m,n from self.idx_c
                    ax.plot(self.t, real(c))#, label=f'rho_{m,n}')
            return ax
                    
        def field_plot(ax, title):
            ax = plot_ax(ax, title)
            for i,f in enumerate(self.fields): # these are lambda functions
                ax.plot(self.t, [real(f(t)) for t in self.t], label=rf'$\Omega${i+1}')
            return ax
                
        def mixing_plot(ax, title):
            ax = plot(ax, title)
            assert len(self.fields) == 2
            f1, f2 = self.fields 
            ax.plot(self.t, [atan(f2(t)/f1(t)) for t in self.t]) # double check this
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
        
        fig, axes = plt.subplots(1, sum([plotdict[key]['show'] for key in plotdict]),
                                **kwargs)
        if type(axes) != ndarray: # only one subplot
            axes = [axes]

        ax_idx = 0
        for propdict in plotdict.values():
            if propdict['show']:
                axes[ax_idx] = propdict['plot_func'](axes[ax_idx], propdict['title'])
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