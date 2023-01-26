#::: imports
import warnings
import numpy as np
import ellc



#::: load data
def load_data(target_name):
        
    if target_name == 'Select a target...':
        return None
        
    elif target_name == 'WASP-189b':
        time, flux, flux_err = np.genfromtxt('WASP-189b_data.csv', comments='#', delimiter=',', unpack=True)
        
        #::: select only the latest observation request (one full transit)
        ind = np.where(time>9018)
        time = time[ind]
        flux = flux[ind]
        flux_err = flux_err[ind]

    elif target_name == 'TOI-560c':
        warnings.warn('The target '+target_name+' is not implemented yet.') #TODO
    
    elif target_name == 'KELT-3b':
        warnings.warn('The target '+target_name+' is not implemented yet.') #TODO

    #::: start counting time from the first observation
    time -= time[0]
        
    #::: define the model's time grid for the plots
    time_model = np.linspace(time[0], time[-1], 1000)
    
    return time, flux, flux_err, time_model


        
#::: load params
def load_params(target_name):
    
    if target_name == 'Select a target...':
        return None, None, None, None, None, None
    
    #https://exoplanetarchive.ipac.caltech.edu/overview/wasp-189
    elif target_name == 'WASP-189b':
        '''
        radius_planet = 18.15 #Earth radii
        radius_star = 2.36 #Solar radii
        epoch = 8926.5416960 #BJD_TDB
        period = 2.7240330 #days
        a = 0.05053 #AU
        incl = 84.03 #deg
        '''
        radius_planet_init = 1. #Earth radii
        radius_star_init = 1. #Solar radii
        epoch_init = 0.3 #BJD_TDB
        period = 2.7240330 #days
        a = 0.05053 #AU -> Solar radii
        incl = 84.03 #deg  
        
    elif target_name == 'TOI-560c':
        warnings.warn('The target '+target_name+' is not implemented yet.') #TODO
    
    elif target_name == 'KELT-3b':
        warnings.warn('The target '+target_name+' is not implemented yet.') #TODO

    else:
        raise ValueError('Oh whoopsie. Something went wrong.')
    
    return radius_planet_init, radius_star_init, epoch_init, period, a, incl



#::: call ellc to create light curves
def calc_flux_model(radius_planet, radius_star, epoch, period, a, incl, time_model):
    
    flux_model = ellc.fluxes(
                            t_obs =       time_model, 
                            radius_1 =    radius_star/(a * 215.03215567054764), #both in Solar radii
                            radius_2 =    radius_planet/(a * 23454.927125633025), #both in Earth radii
                            sbratio =     1e-12, 
                            incl =        incl, #in degree
                            t_zero =      epoch, #in days
                            period =      period, #in days
                            ldc_1 =       0.3, #TODO 0.495
                            ldc_2 =       0.5, #TODO 0.055
                            grid_1 =      'sparse',
                            grid_2 =      'sparse',
                            ld_1 =        'quad',
                            ld_2 =        'quad',
                            verbose =     False
                            )[0]
    
    return flux_model



#::: check if the initial guess is good enough to start the fit
def check_initial_guess(target_name, radius_planet_init, radius_star_init, epoch_init):
    
    if target_name == 'Select a target...':
        return False
    
    #https://exoplanetarchive.ipac.caltech.edu/overview/wasp-189
    elif target_name == 'WASP-189b':    
        if (np.abs(radius_planet_init-18.15) <= 2.) & (np.abs(radius_star_init-2.36) <= 0.2) & (np.abs(epoch_init-0.36) <= 0.01):
            return True
        else:
            return False
        
    elif target_name == 'TOI-560c':
        warnings.warn('The target '+target_name+' is not implemented yet.') #TODO
        return False
    
    elif target_name == 'KELT-3b':
        warnings.warn('The target '+target_name+' is not implemented yet.') #TODO
        return False

    else:
        raise ValueError('Oh whoopsie. Something went wrong.')
        
        

#::: load the bounds        
def load_bounds(target_name):
    
    if target_name == 'Select a target...':
        return None
    
    #https://exoplanetarchive.ipac.caltech.edu/overview/wasp-189
    elif target_name == 'WASP-189b':    
        bounds = [
                  ('normal',18.15,0.24), #radius planet
                  ('normal',2.36,0.03), #radius star
                  ('uniform',0.3,0.4), #epoch
                 ]
        return bounds
        
    elif target_name == 'TOI-560c':
        warnings.warn('The target '+target_name+' is not implemented yet.') #TODO
        return None
    
    elif target_name == 'KELT-3b':
        warnings.warn('The target '+target_name+' is not implemented yet.') #TODO
        return None

    else:
        raise ValueError('Oh whoopsie. Something went wrong.')