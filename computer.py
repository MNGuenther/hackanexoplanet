#::: imports
import warnings
import numpy as np
import ellc



#::: load data
def load_data(target_name):
        
    if target_name == 'Select an exoplanet...':
        return None
        
    elif target_name == 'WASP-189b':
        time, flux, flux_err = np.genfromtxt('data/WASP-189b_data.csv', comments='#', delimiter=',', unpack=True)
        
        #::: select only the latest observation request (one full transit)
        ind = np.where(time>9018.)
        time = time[ind]
        flux = flux[ind]
        flux_err = flux_err[ind]
    
    elif target_name == 'KELT-3b':
        time, flux, flux_err = np.genfromtxt('data/KELT-3b_data.csv', comments='#', delimiter=',', unpack=True)

    elif target_name == 'TOI-560c':
        time, flux, flux_err = np.genfromtxt('data/TOI-560c_data.csv', comments='#', delimiter=',', unpack=True)

    elif target_name == 'K2-141b':
        time, flux, flux_err = np.genfromtxt('data/K2-141b_data.csv', comments='#', delimiter=',', unpack=True)
        
    #::: remove the offset
    time -= time[0]
        
    #::: define the model's time grid for the plots
    time_model = np.linspace(time[0], time[-1], 1000)
    
    return time, flux, flux_err, time_model


        
#::: load params
def load_params(target_name):
    
    if target_name == 'Select an exoplanet...':
        return None, None, None, None, None, None
    
    #https://exoplanetarchive.ipac.caltech.edu/overview/wasp-189
    elif target_name == 'WASP-189b':
        radius_planet_init = 1. #Earth radii
        radius_star_init = 1. #Solar radii
        epoch_init = 0.1 #days since the first data time stamp (i.e. not the BJD_TDB)
        period = 2.7240330 #days; from Stassun+17
        a = 0.05053 #AU -> Solar radii
        incl = 84.03 #deg  
        q1 = 0.181
        q2 = 0.472
    
    #https://exoplanetarchive.ipac.caltech.edu/overview/kelt-3
    elif target_name == 'KELT-3b':
        radius_planet_init = 1. #Earth radii
        radius_star_init = 1. #Solar radii
        epoch_init = 0.1 #days since the first data time stamp (i.e. not the BJD_TDB)
        period = 2.70339 #days; from Stassun+17
        a = 0.0464 #AU -> Solar radii #allesfitter gave 0.0464; Stassun+17 reports 0.0476; Pepper+13 reports 0.04122
        incl = 84.14 #deg; from Stassun+17
        q1 = 0.55 #from allesfitter
        q2 = 0.19 #from allesfitter
        
    #https://exoplanetarchive.ipac.caltech.edu/overview/toi-560
    elif target_name == 'TOI-560c':
        radius_planet_init = 1. #Earth radii
        radius_star_init = 1. #Solar radii
        epoch_init = 0.1 #days since the first data time stamp (i.e. not the BJD_TDB)
        period = 18.87974 #days; from Barragan+19
        a = 0.1242 #AU -> Solar radii #allesfitter gave 0.1245; Barragan+19 reports 0.1242
        incl = 89.72 #deg; from Barragan+19
        q1 = 0.44 #from allesfitter
        q2 = 0.48 #from allesfitter
        
    #https://exoplanetarchive.ipac.caltech.edu/overview/k2-141b
    elif target_name == 'K2-141b':
        radius_planet_init = 1. #Earth radii
        radius_star_init = 1. #Solar radii
        epoch_init = 0.1 #days since the first data time stamp (i.e. not the BJD_TDB)
        period = 0.280324822 #days; from Barragan+19
        a = 0.00747 #AU; from Bonomo+23 and Malavolta+18
        incl = 86.3 #deg; from Bonomo+23 and Malavolta+18
        q1 = 0.5 #fixed initial guess
        q2 = 0.5 #fixed initial guess
        
    else:
        raise ValueError('Oh whoopsie. Something went wrong.')
    
    return radius_planet_init, radius_star_init, epoch_init, period, a, incl, q1, q2



#::: call ellc to create light curves
def calc_flux_model(radius_planet, radius_star, epoch, period, a, incl, q1, q2, time_model):
    
    #try/except and if statements are needed for K2-141b and its super short period
    try:
        flux_model = ellc.fluxes(
                                t_obs =       time_model, 
                                radius_1 =    radius_star/(a * 215.03215567054764), #both in Solar radii
                                radius_2 =    radius_planet/(a * 23454.927125633025), #both in Earth radii
                                sbratio =     1e-12, 
                                incl =        incl, #in degree
                                t_zero =      epoch, #in days
                                period =      period, #in days
                                ldc_1 =       q1,
                                ldc_2 =       q2,
                                grid_1 =      'sparse',
                                grid_2 =      'sparse',
                                ld_1 =        'quad',
                                ld_2 =        'quad',
                                verbose =     False
                                )[0]
    except: 
        flux_model = np.zeros_like(time_model) 
        
    if np.abs(np.nanmedian(flux_model) - 1.) > 1.:
        flux_model = np.zeros_like(time_model)
        
    if epoch > period/2.:
        flux_model = np.zeros_like(time_model)
        
    return flux_model



#::: check if the initial guess is good enough to start the fit
def check_initial_guess(target_name, radius_planet_init, radius_star_init, epoch_init):
    
    if target_name == 'Select an exoplanet...':
        return False
    
    #https://exoplanetarchive.ipac.caltech.edu/overview/wasp-189
    elif target_name == 'WASP-189b':    
        if (np.abs(radius_planet_init-18.15) <= 2.) & (np.abs(radius_star_init-2.36) <= 0.2) & (np.abs(epoch_init-0.36) <= 0.01):
            return True
        else:
            return False
    
    #https://exoplanetarchive.ipac.caltech.edu/overview/kelt-3b
    elif target_name == 'KELT-3b':
        if (np.abs(radius_planet_init-17.5) <= 2.) & (np.abs(radius_star_init-1.70) <= 0.2) & (np.abs(epoch_init-0.27633850202023424) <= 0.02): 
            #to account for "time -= time[0]" above, we need to convert all epochs accordingly
            #thus, the correct epoch = (epoch - time[0])%period = (2457472.49813 - 2.459967450576898176e+06)%2.70338980 = 0.27633850202023424
            return True
        else:
            return False
        
    #https://exoplanetarchive.ipac.caltech.edu/overview/TOI-560c
    elif target_name == 'TOI-560c':
        if (np.abs(radius_planet_init-2.39) <= 1.) & (np.abs(radius_star_init-0.65) <= 0.2) & (np.abs(epoch_init-0.45) <= 0.02):
            #to account for "time -= time[0]" above, we need to convert all epochs accordingly
            #thus, the correct epoch = (epoch - time[0])%period = (2459232.1682 - 2.459968016734779812e+06)%18.87974 = 0.46132522037691714
            return True
        else:
            return False

    #https://exoplanetarchive.ipac.caltech.edu/overview/TOI-560c
    elif target_name == 'K2-141b':
        if (np.abs(radius_planet_init-1.51) <= 0.5) & (np.abs(radius_star_init-0.681) <= 0.2) & (np.abs(epoch_init-0.07) <= 0.01):
            #to account for "time -= time[0]" above, we need to convert all epochs accordingly
            #thus, ...
            return True
        else:
            return False
        
    else:
        raise ValueError('Oh whoopsie. Something went wrong.')
        
        

#::: load the bounds        
def load_bounds(target_name):
    
    if target_name == 'Select an exoplanet...':
        return None
    
    #https://exoplanetarchive.ipac.caltech.edu/overview/wasp-189
    elif target_name == 'WASP-189b':    
        bounds = [
                  ('normal',18.15,0.24), #radius planet
                  ('normal',2.36,0.03), #radius star
                  ('uniform',0.3,0.4), #epoch
                 ]
        return bounds
    
    #https://exoplanetarchive.ipac.caltech.edu/overview/kelt-3b
    elif target_name == 'KELT-3b':  
        bounds = [
                  ('normal',17.5,1.2), #radius planet
                  ('normal',1.70,0.12), #radius star
                  ('uniform',0.25,0.30), #epoch
                  #to account for "time -= time[0]" above, we need to convert all epochs accordingly
                  #thus, the correct epoch = (epoch - time[0])%period = (2457472.49813 - 2.459967450576898176e+06)%2.70338980 = 0.27633850202023424
                 ]
        return bounds
        
    elif target_name == 'TOI-560c':
        bounds = [
                  ('normal',2.39,0.10), #radius planet
                  ('normal',0.65,0.02), #radius star
                  ('uniform',0.4,0.5), #epoch
                  #to account for "time -= time[0]" above, we need to convert all epochs accordingly
                  #thus, the correct epoch = (epoch - time[0])%period = (2459232.1682 - 2.459968016734779812e+06)%18.87974 = 0.46132522037691714
                 ]
        return bounds

    elif target_name == 'K2-141b':
        bounds = [
                  ('normal',1.51,0.05), #radius planet
                  ('normal',0.681,0.018), #radius star
                  ('uniform',0.05,0.09), #epoch
                  #to account for "time -= time[0]" above, we need to convert all epochs accordingly
                  #thus, the correct epoch = (epoch - time[0])%period = (2459232.1682 - 2.459968016734779812e+06)%18.87974 = 0.46132522037691714
                 ]
        return bounds
    
    else:
        raise ValueError('Oh whoopsie. Something went wrong.')