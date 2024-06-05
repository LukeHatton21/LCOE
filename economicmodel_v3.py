import numpy as np
import xarray as xr
import glob
import pandas as pd
import pvlib
import matplotlib.pyplot as plt
import cartopy.crs as ccrs



class Economic_Profile:
    def __init__(self, solar_capex, wind_capex, renew_op_cost, land_foundations, renewables_capacity, turbine_diameter, turbine_rating, lifetime, renew_discount_rate):
        """ Initialises the economic_profile class
       
        Inputs:
        Solar_capex - base assumption for solar capital expenditure, in USD/kW
        Wind_capex - base assumption for wind capital expenditure, in USD/kW
        Renew_op_cost - base assumption for renewables operating costs, as a % of CAPEX per year.
        Renewables_capacity - nominal renewables capacity
        Land_foundations - base assumption for capital expenditure on land foundations for onshore wind, in USD/kW
        Lifetime - number of years of operation
        Turbine_diameter - diameter of modelled turbine, in m
        Turbine_rating - power rating of modelled turbine, in kW
        Renew_discount_rate - default renewable discount rate, in %
        """
          
        
        
        # Read in economic parameters
        self.solar_capex = solar_capex
        self.wind_capex = wind_capex
        self.renew_op_cost = renew_op_cost
        self.land_foundations = land_foundations
        self.renew_discount_rate = renew_discount_rate
        
        
        # Read in assumptions 
        self.renewables_capacity = 1000 # BASIS of 1 MW
        self.turbine_diameter = turbine_diameter
        self.turbine_rating = turbine_rating
        self.lifetime = lifetime

        
    
    
    

    def get_foundation_cost(self, geodata):
        """ Method to calculate the foundation cost for the wind turbine based on the depth of water, 
        using relationships from Bosch et al 2019 """
        
        depth_data = geodata['depth']
        
        # Set up relationships with depth
        a_parameter = [201, 114.24, 0]
        b_parameter = [612.93, -2270, 773.85]
        c_parameter = [411464, 531738, 680651]
        cutoff_data = [0, 25, 55, 1000]

        # initialise an empty array
        foundation_costs = xr.zeros_like(depth_data)

        # Use relationships with depth to estimate the foundation costs
        for i in range(len(cutoff_data) - 1):
            a = a_parameter[i]
            b = b_parameter[i]
            c = c_parameter[i]
            cutoff_start = cutoff_data[i]
            cutoff_end = cutoff_data[i + 1]

            # Apply cost relationship where depth is greater than the cutoff depth and not NaN
            foundation_costs = xr.where((depth_data > cutoff_start) & (depth_data <= cutoff_end), a * depth_data ** 2 + b * depth_data + c, foundation_costs)
        
        # Apply relationships for onshore (set foundation cost to input) and offshore above the cutoff depth (>1000, N/A)
        foundation_costs = foundation_costs / 1000  # convert all into USD/kW
        foundation_costs = xr.where(geodata['offshore'] == True, foundation_costs, self.land_foundations)
        foundation_costs = xr.where(depth_data > 1000, np.nan, foundation_costs)
        foundation_costs = foundation_costs * self.renewables_capacity  # In USD for 1 MW

        return foundation_costs
    
    def get_transmission_cost(self, dist_data):
        """ Method to calculate the cost of electricity transmission (either through HVAC or HVDC) to shore, using
        relationships from the International Energy Agency's Wind Energy Outlook 2019 """
        dist = dist_data
        
        # Initialise empty arrays
        hvac = xr.zeros_like(dist)
        hvdc = xr.zeros_like(dist)      
        
        # Apply IEA relationships
        hvac = xr.where(dist > 0, (0.0085 * dist + 0.0568), 0) * 1000 # Conversion to USD/kW
        hvdc = xr.where(dist > 0, (0.0022 * dist + 0.3878), 0) * 1000 # Conversion to USD/kW
        transmission_costs = np.minimum(hvac, hvdc)
        transmission_costs = transmission_costs * self.renewables_capacity #  # In USD for 1 MW
        
        return transmission_costs
    
    
    def get_interarray_costs(self, technology):
        """ Method to calculate the inter-array distance between wind turbines at each location and calculate the 
        cost of AC cables between all of the wind turbines """
        
        # Get installed wind capacity
        wind_capacity = self.renewables_capacity
        turbine_rating = self.turbine_rating
        turbine_diameter = self.turbine_diameter
        
        # Calculate number of turbines
        n_turbines = wind_capacity / turbine_rating
        
        # Calculate interarray distance
        spacing = 7.5 * turbine_diameter / 1000 
        interarray_dist = n_turbines * spacing
        interarray_cost = (0.0085 * interarray_dist + 0.0568) * 1000 * turbine_rating
        
        return interarray_cost
    
    
    def offshore_generation(self, geodata):
        

        # Read out the geodata
        dist_data = geodata['distance']
        depth = geodata['depth']
        offshore = geodata['offshore']
        
        # Create a storage vector                           
        offshore_costs = xr.zeros_like(dist_data)
        
        # Get transmission costs
        transmission_costs = self.get_transmission_cost(dist_data)
        
        # Calculate the interarray cable costs
        interarray_cable_costs = self.get_interarray_costs('AC')
        
        # Calculate the offshore substation costs taken from https://guidetoanoffshorewindfarm.com/wind-farm-costs 
        # and including the offshore substation total cost + installation cost
        offshore_substation_costs = 155 * 1.25 * self.renewables_capacity  # In USD for 1 MW
            
        # Calculate the foundation costs
        foundation_costs = self.get_foundation_cost(geodata)
        

        # Set costs equal to zero for onshore locations
        foundation_costs = xr.where(offshore == True, foundation_costs, np.nan)
        transmission_costs = xr.where(offshore == True, transmission_costs, np.nan)
        interarray_cable_costs = xr.where(offshore == True, interarray_cable_costs, np.nan)
        other_costs = xr.where(offshore == True, offshore_substation_costs, np.nan)
        
        # Create an xarray dataset
        data_vars = {'foundation_costs': foundation_costs,
            'transmission_costs': transmission_costs,
        'interarray_costs': interarray_cable_costs,
        'other_costs': xr.full_like(dist_data, offshore_substation_costs)}
        coords = {'latitude': dist_data.latitude,'longitude': dist_data.longitude}

        offshore_cost_breakdown = xr.Dataset(data_vars=data_vars, coords=coords)  # In USD for 1 MW
        
                                   
        return offshore_cost_breakdown
    

        
        
        
        
        
    def configuration_analysis(self, geodata):
    
        # Get offshore data from Geodata
        offshore = geodata['offshore']
        
        
        # Sum the costs of turbine, transmission and foundation 
        turbine_foundation_costs = self.land_foundations * self.renewables_capacity  # In USD for 1 MW
        wind_turbine_costs = self.wind_capex * self.renewables_capacity  # In USD for 1 MW
        
        # Extract the total cost for each of the locations
        offshore_breakdown = self.offshore_generation(geodata)
    
        
        # Sum up costs for onshore and offshore
        onshore_costs = wind_turbine_costs + turbine_foundation_costs  # In USD for 1 MW
        offshore_costs = offshore_breakdown['foundation_costs'] + offshore_breakdown['transmission_costs'] + offshore_breakdown['interarray_costs'] + offshore_breakdown['other_costs'] + wind_turbine_costs  # In USD for 1 MW
        
        # Calculate the costs for onshore and offshore
        foundation_costs = xr.where(offshore == True, offshore_breakdown['foundation_costs'], turbine_foundation_costs)
        total_offshore_costs = xr.where(offshore == True, offshore_costs, np.nan)
        total_offshore_costs = xr.where(geodata['depth'] < 1000, total_offshore_costs, np.nan)
        total_onshore_costs = xr.where(offshore == True, np.nan, onshore_costs)

        total_costs = xr.where(offshore == True, total_offshore_costs, total_onshore_costs)
        
        # Create a dataset with the three possible capital expenditures 
        data_vars = {'wind_costs': total_costs, 'turbine_costs' : wind_turbine_costs, 'foundation_costs': foundation_costs,'transmission_costs': offshore_breakdown['transmission_costs'], "interarray_costs": offshore_breakdown['interarray_costs'], "other_costs": offshore_breakdown['other_costs']}
        coords = {'latitude': geodata.latitude,
                  'longitude': geodata.longitude}
        capital_costs = xr.Dataset(data_vars=data_vars, coords=coords)
        
        
        # Return the dataset
        return capital_costs
        

    

    
    def locational_operating_costs(self, capital_costs, renewables_data_yearly):
        """Calculates the operating cost associated with the renewable or electrolyser capacity as a % of the CAPEX at each location"""
        # Calculate operating cost as a proportion of CAPEX
        operating_cost = capital_costs * self.renew_op_cost 
    
        # Broadcast operating_cost to match the shape of renewables_data_yearly
        operating_costs_extended = operating_cost.broadcast_like(renewables_data_yearly)

        return operating_costs_extended
        
        
        
    
    
    def plot_data(self, data, name):
    
        # Set up data
        latitudes = data.latitude.values
        longitudes = data.longitude.values
        values = data.values

        # create the heatmap using pcolormesh
        fig = plt.figure(figsize=(20, 10))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())
        heatmap = ax.pcolormesh(longitudes, latitudes, values, transform=ccrs.PlateCarree(), cmap='plasma')
        fig.colorbar(heatmap, ax=ax, shrink=0.5)


        # set the extent and aspect ratio of the plot
        ax.set_extent([longitudes.min(), longitudes.max(), latitudes.min(), latitudes.max()], crs=ccrs.PlateCarree())
        aspect_ratio = (latitudes.max() - latitudes.min()) / (longitudes.max() - longitudes.min())
        ax.set_aspect(aspect_ratio)

        # add axis labels and a title
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title(name + ' heatmap')
        ax.coastlines()
        ax.stock_img()
        
        plt.show()

    
    
  
    
    
    
    def calculate_costs_and_output_OLD(model, solar_profile, wind_profile, geodata, solar_fraction):
        
        
        # Specify renewable output
        renewables_data = solar_profile * solar_fraction + wind_profile * (1 - solar_fraction)   
        renewables_data_yearly = renewables_data.groupby('time.year').sum(dim='time')
        

        
        # Extract dimensions from the renewables_profile
        years = renewables_data_yearly.year
        latitudes = renewables_data_yearly.latitude.values
        longitudes = renewables_data_yearly.longitude.values 


        # Need to account for CAPEX only in the 0th year
        lat_len = renewables_data_yearly.latitude.size
        lon_len = renewables_data_yearly.longitude.size
        years_len = len(years)
        new_year = [years[0]-1]
        years_appended = np.concatenate((new_year, years))

        
        # Create new arrays for storage
        solar_costs_array = xr.DataArray(dims=('year', 'latitude', 'longitude'),
                                        coords={'year': new_year,
                                                'latitude': [latitudes],
                                                 'longitude': [longitudes]})
        wind_costs_array = xr.DataArray(dims=('year', 'latitude', 'longitude'),
                                        coords={'year': new_year,
                                                'latitude': [latitudes],
                                                 'longitude': [longitudes]})
        land_costs_array = xr.DataArray(dims=('year', 'latitude', 'longitude'),
                                        coords={'year': new_year,
                                                'latitude': [latitudes],
                                                 'longitude': [longitudes]})

        
        
        # Calculate solar capital costs using depth
        solar_capital_costs = xr.zeros_like(geodata['land']) + self.solar_capex * self.renewables_capacity
        
        # Calculate wind capital costs using depth
        wind_costs = self.configuration_analysis(geodata)
        
        # Check if lat_len and lon_len are 1
        if lat_len == 1 & lon_len == 1:

            # Set latitude and longitude as dimensions
            renewables_data_yearly = renewables_data_yearly.expand_dims(latitude=[latitudes], longitude=[longitudes])
            wind_costs = wind_costs.expand_dims(latitude=[latitudes], longitude=[longitudes])


         
        # Set capital costs to zero if solar fraction is 1 or 0
        if solar_fraction == 1:
            wind_costs = xr.zeros_like(wind_costs)
        if solar_fraction == 0:
            solar_capital_costs = xr.zeros_like(solar_capital_costs)
            
        # Extract wind capital costs
        wind_capital_costs = wind_costs['wind_costs']
        
        # Set land costs
        land_capital_costs = 0.5
        ## NEED TO ADAPT TO REMOVE LAND AS A TIME VARYING COST AND INCLUDE A FUNCTION ##
        
        
        # Transfer capital costs across to the relevant cost arrays
        solar_costs_array[0, :, :] = solar_capital_costs * solar_fraction
        wind_costs_array[0, :, :] = wind_capital_costs * (1 - solar_fraction)
        land_costs_array[0, :, :] = land_capital_costs     
        
        # Calculate the operating costs 
        solar_op_costs_array = xr.DataArray(self.locational_operating_costs(solar_capital_costs, renewables_data_yearly), dims=('year', 'latitude', 'longitude'), coords={'year': years,'latitude': [latitudes],'longitude': [longitudes]})
        wind_op_costs_array = xr.DataArray(self.locational_operating_costs(wind_capital_costs, renewables_data_yearly), dims=('year', 'latitude', 'longitude'), coords={'year': years,'latitude': [latitudes],'longitude': [longitudes]})
        land_op_costs_array = xr.zeros_like(wind_op_costs_array)
        

        
        # Combine capital and operating cost arrays
        solar_costs_combined = xr.concat([solar_costs_array, solar_op_costs_array], dim='year')
        wind_costs_combined = xr.concat([wind_costs_array, wind_op_costs_array], dim='year')
        land_costs_combined = xr.concat([land_costs_array, land_op_costs_array], dim='year')
        total_costs_array = solar_costs_combined + wind_costs_combined + land_costs_combined
        renewable_electricity_data = xr.concat([land_costs_array * 0, renewables_data_yearly], dim = 'year')
    
        
        
        # Create a dataset with all the arrays
        data_vars = {'renewable_electricity': renewable_electricity_data,
                     'total_costs': total_costs_array,
                     'solar_costs': solar_costs_combined, 
                     'wind_costs': wind_costs_combined,
                     'land_costs': land_costs_combined, 
            **wind_costs.data_vars}               
        coords = {'year': years_appended,
                  'latitude': latitudes,
                  'longitude': longitudes}
        combined_data = xr.Dataset(data_vars=data_vars, coords=coords)
        return combined_data
    
    
    
    def calculate_costs_and_output(self, geodata, solar_profile, wind_profile, solar_fraction):
        
        
        # Specify renewable output
        renewables_data = solar_profile * solar_fraction + wind_profile * (1 - solar_fraction) # In KWh for 1 MW
        renewables_data_yearly = renewables_data.groupby('time.year').sum(dim='time') # In KWh for 1 MW
        

        
        # Extract dimensions from the renewables_profile
        years = renewables_data_yearly.year
        latitudes = renewables_data_yearly.latitude
        longitudes = renewables_data_yearly.longitude

        # Need to account for CAPEX only in the 0th year
        years_len = len(years)
        new_year = [years[0]-1]
        years_appended = np.concatenate((new_year, years))

        
        # Create new arrays for storage
        solar_costs_array = xr.DataArray(dims=('year', 'latitude', 'longitude'),
                                        coords={'year': new_year,
                                                'latitude': latitudes,
                                                 'longitude': longitudes})
        wind_costs_array = xr.DataArray(dims=('year', 'latitude', 'longitude'),
                                        coords={'year': new_year,
                                                'latitude': latitudes,
                                                 'longitude': longitudes})
        land_costs_array = xr.DataArray(dims=('year', 'latitude', 'longitude'),
                                        coords={'year': new_year,
                                                'latitude': latitudes,
                                                 'longitude': longitudes})

        
        
        # Calculate solar capital costs using depth
        solar_capital_costs = xr.zeros_like(geodata['land']) + self.solar_capex * self.renewables_capacity # In USD for 1 MW
        
        # Calculate wind capital costs using depth
        wind_costs = self.configuration_analysis(geodata)
        wind_costs.expand_dims(year=new_year)
        # Check if lat_len and lon_len are 1
        #if lat_len == 1 & lon_len == 1:

            # Set latitude and longitude as dimensions
            #print(wind_costs)
            #wind_costs = wind_costs.expand_dims(latitude=latitudes, longitude=longitudes)


         
        # Set capital costs to zero if solar fraction is 1 or 0
        if solar_fraction == 1:
            wind_costs = xr.zeros_like(wind_costs)
        if solar_fraction == 0:
            solar_capital_costs = xr.zeros_like(solar_capital_costs)
            
        # Extract wind capital costs
        wind_capital_costs = wind_costs['wind_costs']
        
        # Set land costs
        land_capital_costs = 0.5
        ## NEED TO ADAPT TO REMOVE LAND AS A TIME VARYING COST AND INCLUDE A FUNCTION ##
        
        
        # Transfer capital costs across to the relevant cost arrays
        solar_costs_array[0, :, :] = solar_capital_costs * solar_fraction
        wind_costs_array[0, :, :] = wind_capital_costs * (1 - solar_fraction)
        land_costs_array[0, :, :] = land_capital_costs        
        
        # Calculate the operating costs 
        solar_op_costs_array = xr.DataArray(self.locational_operating_costs(solar_capital_costs, renewables_data_yearly), dims=('year', 'latitude', 'longitude'), coords={'year': years,'latitude': latitudes,'longitude': longitudes})
        wind_op_costs_array = xr.DataArray(self.locational_operating_costs(wind_capital_costs, renewables_data_yearly), dims=('year', 'latitude', 'longitude'), coords={'year': years,'latitude': latitudes,'longitude': longitudes})
        land_op_costs_array = xr.zeros_like(wind_op_costs_array)
        

        
        # Combine capital and operating cost arrays
        solar_costs_combined = xr.concat([solar_costs_array, solar_op_costs_array], dim='year')
        wind_costs_combined = xr.concat([wind_costs_array, wind_op_costs_array], dim='year')
        land_costs_combined = xr.concat([land_costs_array, land_op_costs_array], dim='year')
        total_costs_array = solar_costs_combined + wind_costs_combined + land_costs_combined
        renewable_electricity_data = xr.concat([land_costs_array * 0, renewables_data_yearly], dim = 'year')
        wind_costs = wind_costs.drop("wind_costs")
        
        
        # Create a dataset with all the arrays
        data_vars = {'renewable_electricity': renewable_electricity_data,
                     'total_costs': total_costs_array,
                     'solar_costs': solar_costs_combined, 
                     'wind_costs': wind_costs_combined,
                     'land_costs': land_costs_combined, 
            **wind_costs.data_vars}               
        coords = {'year': years_appended,
                  'latitude': latitudes,
                  'longitude': longitudes}
        combined_data = xr.Dataset(data_vars=data_vars, coords=coords)
        return combined_data
    
    
    def calculate_revenue(self, geodata, solar_profile, wind_profile, solar_fraction, electricity_prices, return_prices=None, five_years=None):
        
        
        # Specify renewable output
        renewables_data = solar_profile * solar_fraction + wind_profile * (1 - solar_fraction) # IN kWH
        renewables_data_yearly = renewables_data.groupby('time.year').sum(dim='time')
        
        # Extract dimensions from the renewables_profile
        years = renewables_data_yearly.year.values
        latitudes = renewables_data_yearly.latitude
        longitudes = renewables_data_yearly.longitude
        years_list = [str(year) for year in years]
        
        # Take 5 year average if only one year of generation data
        def calculate_date_without_year(x):
            if x.month == 2 and x.day == 29:
                return pd.NaT  # Return NaT for February 29th
            else:
                return pd.Timestamp(year=int(years_list[0]), month=x.month, day=x.day, hour=x.hour)
            
        if len(years_list) == 1:
            current_year = np.min(years)
            previous_years = np.arange(current_year - 4, current_year + 1)

            # Convert the years to strings
            previous_years_str = [str(year) for year in previous_years]
            
            # Select electricity prices for the last five years
            electricity_yearly_prices_last5years = electricity_prices.sel(time=slice(previous_years_str[0], previous_years_str[-1]))
            

            df = electricity_yearly_prices_last5years.to_dataframe()
            # Extract month and day from the datetime index
            df['month'] = df.index.get_level_values('time').month
            df['day'] = df.index.get_level_values('time').day
            
            df['new_time'] = df.index.get_level_values('time').map(calculate_date_without_year)

            # Group by latitude, longitude, and the datetime minus the year, then calculate the mean
            mean_prices_last5years = df.groupby(['latitude', 'longitude', 'new_time'])['Price (USD/kWh)'].mean()
            mean_prices = mean_prices_last5years.to_xarray()
            electricity_yearly_prices = mean_prices.rename({"new_time": "time"})
            self.electricity_yearly_prices_store = electricity_yearly_prices
            price_data_resampled = electricity_yearly_prices.resample(time='30T').ffill()
            electricity_revenue_5_years = price_data_resampled * renewables_data
            
        else:
            electricity_revenue = electricity_prices.sel(time=slice(years_list[0], years_list[-1])) * renewables_data
        
        # Multiple electricity prices by renewables data to get profit
        if len(years_list) == 1:
            # Subtract one year from the first year
            first_year = np.min(years)
            previous_year = first_year - 1
            previous_year_str = str(previous_year)
            electricity_yearly_prices = electricity_prices.sel(time=previous_year_str)

            # Add one year to the prices datetime string
            times =electricity_yearly_prices['time'].values
            time_pd = pd.to_datetime(times)
            new_time_pd = time_pd + pd.Timedelta(days=365)
            new_times = new_time_pd.values
            electricity_yearly_prices['time'] = new_times
            self.electricity_yearly_prices = electricity_yearly_prices
            self.renewables_data = renewables_data
            
            if electricity_yearly_prices.time.size == 8759:
                price_data_resampled = electricity_yearly_prices.resample(time='30T').ffill()
                electricity_yearly_prices = price_data_resampled
                
            # Calculate revenue
            electricity_revenue = electricity_yearly_prices * renewables_data # IN USD/kWH * KWh
            prices = electricity_yearly_prices
        else:
            prices = electricity_prices.sel(time=slice(years_list[0], years_list[-1])) 
            electricity_revenue = prices * renewables_data # IN USD/kWH * KWh
        
        if five_years is not None:
            electricity_revenue = electricity_revenue_5_years
        else:
            electricity_revenue = electricity_revenue
        
        # Sum up revenue by year
        revenue_yearly = electricity_revenue.groupby('time.year').sum(dim='time') # IN USD for 1 MW
        mean_prices_yearly = prices.groupby('time.year').mean(dim='time')
        
        # Need to account for CAPEX only in the 0th year
        years_len = len(years)
        new_year = [years[0]-1]
        years_appended = np.concatenate((new_year, years))

        
        # Create new arrays for storage
        revenue_array = xr.DataArray(revenue_yearly*0,dims=('year', 'latitude', 'longitude'),
                                        coords={'year': new_year,
                                                'latitude': latitudes,
                                                 'longitude': longitudes})
        
        # Concatenate revenue
        revenue_combined = xr.concat([revenue_array, revenue_yearly], dim='year')
        prices_combined = xr.concat([revenue_array, mean_prices_yearly], dim='year')
        
        return revenue_combined, prices_combined
    
    
    
    def calculate_revenue_5_years(model, geodata, solar_profile, wind_profile, solar_fraction, electricity_prices):
        
        
        # Specify renewable output
        renewables_data = solar_profile * solar_fraction + wind_profile * (1 - solar_fraction) # IN kWH
        renewables_data_yearly = renewables_data.groupby('time.year').sum(dim='time')
        
        # Extract dimensions from the renewables_profile
        years = renewables_data_yearly.year.values
        latitudes = renewables_data_yearly.latitude
        longitudes = renewables_data_yearly.longitude
        years_list = [str(year) for year in years]
        
        
            
        if len(years_list) == 1:
            
            # Calculate the number of years you want to extend to
            desired_years = 5
            current_year = np.min(years)
                                  


            # Calculate the number of duplications needed
            num_duplicates = desired_years / 1

            # Duplicate the production data along the time dimension
            extended_production_data = xr.concat([renewables_data]*int(num_duplicates), dim='time')
        
            # Create time period and extract price data
            start_year = current_year - desired_years + 1
            end_year = current_year
            electricity_prices_5_years = electricity_prices.sel(time=slice(str(start_year), str(end_year)))
            price_data_resampled = electricity_prices_5_years.resample(time='30T').ffill()

        
            # Create time range                     
            time_range = pd.date_range(start=f'01-01-{start_year} 00:30:00', end=f'12-31-{end_year} 23:59:59', freq='H')
            time_range_without_feb_29 = time_range[~((time_range.month == 2) & (time_range.day == 29))]
            extended_production_data['time'] = time_range_without_feb_29
            electricity_revenue = price_data_resampled * extended_production_data
            prices = price_data_resampled
        else:
            prices = electricity_prices.sel(time=slice(years_list[0], years_list[-1]))
            electricity_revenue = prices * renewables_data
            

        
        # Sum up revenue by year
        revenue_yearly = electricity_revenue.groupby('time.year').sum(dim='time') # IN USD for 1 MW
        mean_prices_yearly = prices.groupby('time.year').mean(dim='time')
    
        # Need to account for CAPEX only in the 0th year
        years_len = len(years)
        if len(years_list) == 1:
            new_year = current_year - desired_years + 1
        else:
            new_year = years[0] - 1
    
        
        # Create new arrays for storage
        revenue_array = xr.zeros_like(renewables_data_yearly.sel(year=years[0]))
        revenue_array = revenue_array.assign_coords(year=int(new_year))
    
    
        # Concatenate revenue
        revenue_combined = xr.concat([revenue_array, revenue_yearly], dim='year')
        prices_combined = xr.concat([revenue_array, mean_prices_yearly], dim='year')
        
        return revenue_combined, prices_combined
        
        
        